use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::quote;
use syn::{Expr, FnArg, ImplItem, ItemFn, ItemImpl, Lit, Meta, Pat, Type, parse_macro_input};

fn extract_doc(attrs: &[syn::Attribute]) -> Vec<String> {
    attrs
        .iter()
        .filter_map(|attr| {
            if !attr.path().is_ident("doc") {
                return None;
            }
            if let Meta::NameValue(nv) = &attr.meta
                && let Expr::Lit(el) = &nv.value
                && let Lit::Str(s) = &el.lit
            {
                return Some(s.value().trim().to_string());
            }
            None
        })
        .collect()
}

fn parse_doc(lines: &[String]) -> (String, std::collections::HashMap<String, String>) {
    let mut desc_lines = vec![];
    let mut params: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut current_param: Option<String> = None;

    for line in lines {
        if line.is_empty() {
            continue;
        }

        if let Some(key) = current_param.as_ref().filter(|_| {
            line.starts_with(' ')
                || line.starts_with('\t')
                || line.starts_with('-')
                || line.starts_with('•')
        }) {
            let entry = params.entry(key.clone()).or_default();
            entry.push(' ');
            entry.push_str(line.trim());
            continue;
        }

        if let Some((key, val)) = line.split_once(':') {
            let key = key.trim().to_string();
            let val = val.trim().to_string();
            if key.chars().all(|c| c.is_alphanumeric() || c == '_') && !val.is_empty() {
                params.insert(key.clone(), val);
                current_param = Some(key);
                continue;
            }
        }

        current_param = None;
        desc_lines.push(line.clone());
    }
    (desc_lines.join(" ").trim().to_string(), params)
}

fn is_option(ty: &Type) -> bool {
    if let Type::Path(tp) = ty
        && let Some(seg) = tp.path.segments.last()
    {
        return seg.ident == "Option";
    }
    false
}

struct ToolMethod {
    tool_name: String,
    description: String,
    params: Vec<ParamInfo>,
    body: syn::Block,
    output: syn::ReturnType,
}

struct ParamInfo {
    name: String,
    ty: Type,
    desc: String,
    optional: bool,
}

// ── #[tool] (One-shot) ────────────────────────────────────────────────────────

#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    if let Ok(item_fn) = syn::parse::<ItemFn>(item.clone())
        && item_fn.sig.asyncness.is_some()
    {
        return tool_from_fn(attr, item_fn, false);
    }
    tool_from_impl(attr, item, false)
}

// ── #[streaming_tool] ─────────────────────────────────────────────────────────

#[proc_macro_attribute]
pub fn streaming_tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    if let Ok(item_fn) = syn::parse::<ItemFn>(item.clone()) {
        return tool_from_fn(attr, item_fn, true);
    }
    tool_from_impl(attr, item, true)
}

// ── Generators ────────────────────────────────────────────────────────────────

fn tool_from_fn(attr: TokenStream, item_fn: ItemFn, is_streaming: bool) -> TokenStream {
    let fn_name = item_fn.sig.ident.to_string();
    let struct_ident = item_fn.sig.ident.clone();

    let override_name: Option<String> = if !attr.is_empty() {
        let s = TokenStream2::from(attr).to_string();
        s.find('"').and_then(|start| {
            s.rfind('"')
                .filter(|&end| end > start)
                .map(|end| s[start + 1..end].to_string())
        })
    } else {
        None
    };

    let tool_name = override_name.unwrap_or_else(|| fn_name.clone());
    let doc_lines = extract_doc(&item_fn.attrs);
    let (description, param_docs) = parse_doc(&doc_lines);

    let mut params = vec![];
    for arg in &item_fn.sig.inputs {
        if let FnArg::Typed(pt) = arg {
            let name = if let Pat::Ident(pi) = &*pt.pat {
                pi.ident.to_string()
            } else {
                continue;
            };
            let ty = (*pt.ty).clone();
            let desc = param_docs.get(&name).cloned().unwrap_or_default();
            let optional = is_option(&ty);
            params.push(ParamInfo {
                name,
                ty,
                desc,
                optional,
            });
        }
    }

    let method = ToolMethod {
        tool_name,
        description,
        params,
        body: *item_fn.block,
        output: item_fn.sig.output,
    };

    let raw_tools_body = generate_raw_tools(&method);
    let call_arm = generate_call_arm(&method, is_streaming);

    let expanded = quote! {
        #[allow(non_camel_case_types)]
        pub struct #struct_ident;

        #[agentix::async_trait::async_trait]
        impl agentix::tool_trait::Tool for #struct_ident {
            fn raw_tools(&self) -> Vec<agentix::raw::shared::ToolDefinition> {
                vec![#raw_tools_body]
            }

            async fn call(&self, name: &str, args: agentix::serde_json::Value) -> agentix::futures::stream::BoxStream<'static, agentix::tool_trait::ToolOutput> {
                match name {
                    #call_arm
                    _ => {
                        let err = agentix::serde_json::json!({"error": format!("unknown tool: {}", name)});
                        use agentix::futures::StreamExt;
                        agentix::futures::stream::iter(vec![agentix::tool_trait::ToolOutput::Result(err)]).boxed()
                    }
                }
            }
        }
    };

    expanded.into()
}

fn tool_from_impl(attr: TokenStream, item: TokenStream, is_streaming: bool) -> TokenStream {
    let item_impl = parse_macro_input!(item as ItemImpl);

    let override_name: Option<String> = if !attr.is_empty() {
        let s = TokenStream2::from(attr).to_string();
        s.find('"').and_then(|start| {
            s.rfind('"')
                .filter(|&end| end > start)
                .map(|end| s[start + 1..end].to_string())
        })
    } else {
        None
    };

    let mut tool_methods: Vec<ToolMethod> = vec![];

    for item in &item_impl.items {
        if let ImplItem::Fn(method) = item {
            if !is_streaming && method.sig.asyncness.is_none() {
                continue;
            }
            let fn_name = method.sig.ident.to_string();
            let tool_name = override_name.clone().unwrap_or_else(|| fn_name.clone());
            let doc_lines = extract_doc(&method.attrs);
            let (description, param_docs) = parse_doc(&doc_lines);

            let mut params = vec![];
            for arg in &method.sig.inputs {
                if let FnArg::Typed(pt) = arg {
                    let name = if let Pat::Ident(pi) = &*pt.pat {
                        pi.ident.to_string()
                    } else {
                        continue;
                    };
                    let ty = (*pt.ty).clone();
                    let desc = param_docs.get(&name).cloned().unwrap_or_default();
                    let optional = is_option(&ty);
                    params.push(ParamInfo {
                        name,
                        ty,
                        desc,
                        optional,
                    });
                }
            }
            tool_methods.push(ToolMethod {
                tool_name,
                description,
                params,
                body: method.block.clone(),
                output: method.sig.output.clone(),
            });
        }
    }

    let raw_tools_body = tool_methods.iter().map(generate_raw_tools);
    let call_arms = tool_methods.iter().map(|m| generate_call_arm(m, is_streaming));

    let self_ty = &item_impl.self_ty;

    let expanded = quote! {
        #[agentix::async_trait::async_trait]
        impl agentix::tool_trait::Tool for #self_ty {
            fn raw_tools(&self) -> Vec<agentix::raw::shared::ToolDefinition> {
                vec![#(#raw_tools_body),*]
            }

            async fn call(&self, name: &str, args: agentix::serde_json::Value) -> agentix::futures::stream::BoxStream<'static, agentix::tool_trait::ToolOutput> {
                match name {
                    #(#call_arms)*
                    _ => {
                        let err = agentix::serde_json::json!({"error": format!("unknown tool: {}", name)});
                        use agentix::futures::StreamExt;
                        agentix::futures::stream::iter(vec![agentix::tool_trait::ToolOutput::Result(err)]).boxed()
                    }
                }
            }
        }
    };

    expanded.into()
}

fn generate_raw_tools(method: &ToolMethod) -> proc_macro2::TokenStream {
    let tool_name = &method.tool_name;
    let description = &method.description;
    let prop_inserts = method.params.iter().map(|p| {
        let pname = &p.name;
        let pdesc = &p.desc;
        let ty = &p.ty;
        quote! {{
            let schema = <#ty as agentix::schemars::JsonSchema>::json_schema(&mut __gen);
            let mut prop = agentix::serde_json::to_value(schema).unwrap();
            if let Some(obj) = prop.as_object_mut() {
                obj.insert("description".to_string(), agentix::serde_json::json!(#pdesc));
            }
            properties.insert(#pname.to_string(), prop);
        }}
    });
    let required: Vec<&str> = method
        .params
        .iter()
        .filter(|p| !p.optional)
        .map(|p| p.name.as_str())
        .collect();
    
    quote! {{
        let mut __gen = agentix::schemars::SchemaGenerator::default();
        let mut properties = agentix::serde_json::Map::new();
        #(#prop_inserts)*

        let required: Vec<&str> = vec![#(#required),*];
        let mut parameters = agentix::serde_json::json!({
            "type": "object",
            "properties": properties,
            "required": required,
        });

        let defs = __gen.take_definitions();
        if !defs.is_empty() {
            parameters["$defs"] = agentix::serde_json::to_value(defs).unwrap();
        }

        agentix::raw::shared::ToolDefinition {
            kind: agentix::raw::shared::ToolKind::Function,
            function: agentix::raw::shared::FunctionDefinition {
                name: #tool_name.to_string(),
                description: Some(#description.to_string()),
                parameters,
                strict: None,
            },
        }
    }}
}

fn generate_call_arm(method: &ToolMethod, is_streaming: bool) -> proc_macro2::TokenStream {
    let tool_name = &method.tool_name;
    let body = &method.body;
    let output = &method.output;
    let arg_parses = method.params.iter().map(|p| {
        let pname = syn::Ident::new(&p.name, Span::call_site());
        let pname_str = &p.name;
        let ty = &p.ty;
        quote! {
            let #pname: #ty = match agentix::serde_json::from_value(
                args.get(#pname_str).cloned().unwrap_or(agentix::serde_json::Value::Null)
            ) {
                Ok(v) => v,
                Err(e) => {
                    let err = agentix::serde_json::json!({
                        "error": format!("invalid argument '{}': {}", #pname_str, e)
                    });
                    use agentix::futures::StreamExt;
                    return agentix::futures::stream::iter(vec![agentix::tool_trait::ToolOutput::Result(err)]).boxed();
                }
            };
        }
    });

    if is_streaming {
        quote! {
            #tool_name => {
                use agentix::futures::StreamExt;
                #(#arg_parses)*
                
                let __stream = (move || { #body })();
                __stream.boxed()
            }
        }
    } else {
        quote! {
            #tool_name => {
                use agentix::futures::StreamExt;
                #(#arg_parses)*
                
                let __result = (async move || #output { #body })().await;

                #[allow(unused_imports)]
                use agentix::tool_trait::{ToolResultResult, ToolResultValue};

                let __val = (__result).__agentix_wrap();
                agentix::futures::stream::iter(vec![agentix::tool_trait::ToolOutput::Result(__val)]).boxed()
            }
        }
    }
}
