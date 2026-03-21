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
        // Blank lines are ignored.
        if line.is_empty() {
            continue;
        }

        // Indented lines or list markers are continuations of the previous param.
        if current_param.is_some()
            && (line.starts_with(' ')
                || line.starts_with('\t')
                || line.starts_with('-')
                || line.starts_with('•'))
        {
            let key = current_param.as_ref().unwrap();
            let entry = params.entry(key.clone()).or_default();
            entry.push(' ');
            entry.push_str(line.trim());
            continue;
        }

        // `identifier: description` → param entry.
        if let Some((key, val)) = line.split_once(':') {
            let key = key.trim().to_string();
            let val = val.trim().to_string();
            if key.chars().all(|c| c.is_alphanumeric() || c == '_') && !val.is_empty() {
                params.insert(key.clone(), val);
                current_param = Some(key);
                continue;
            }
        }

        // Everything else goes into the function description.
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
}

struct ParamInfo {
    name: String,
    ty: Type,
    desc: String,
    optional: bool,
}

#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    // 先尝试解析为独立 async fn
    if let Ok(item_fn) = syn::parse::<ItemFn>(item.clone())
        && item_fn.sig.asyncness.is_some()
    {
        return tool_from_fn(attr, item_fn);
    }
    // 否则走 impl 块路径
    tool_from_impl(attr, item)
}

fn tool_from_fn(attr: TokenStream, item_fn: ItemFn) -> TokenStream {
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
    };

    let raw_tools_body = {
        let tool_name = &method.tool_name;
        let description = &method.description;
        let prop_inserts = method.params.iter().map(|p| {
            let pname = &p.name;
            let pdesc = &p.desc;
            let ty = &p.ty;
            quote! {{
                let schema = <#ty as agentix::schemars::JsonSchema>::json_schema(&mut __gen);
                let mut prop = serde_json::to_value(schema).unwrap();
                if let Some(obj) = prop.as_object_mut() {
                    obj.insert("description".to_string(), serde_json::json!(#pdesc));
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
            let mut properties = serde_json::Map::new();
            #(#prop_inserts)*

            let required: Vec<&str> = vec![#(#required),*];
            let mut parameters = serde_json::json!({
                "type": "object",
                "properties": properties,
                "required": required,
            });

            // 如果引入了复杂类型，提取 definitions 并注入到 $defs
            let defs = __gen.take_definitions();
            if !defs.is_empty() {
                parameters["$defs"] = serde_json::to_value(defs).unwrap();
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
    };

    let call_arm = {
        let tool_name = &method.tool_name;
        let body = &method.body;
        let arg_parses = method.params.iter().map(|p| {
            let pname = syn::Ident::new(&p.name, Span::call_site());
            let pname_str = &p.name;
            let ty = &p.ty;
            quote! {
                let #pname: #ty = match serde_json::from_value(
                    args.get(#pname_str).cloned().unwrap_or(serde_json::Value::Null)
                ) {
                    Ok(v) => v,
                    Err(e) => return serde_json::json!({
                        "error": format!("invalid argument '{}': {}", #pname_str, e)
                    }),
                };
            }
        });
        quote! {
            #tool_name => {
                #(#arg_parses)*
                let __result = (async move || { #body })().await;
                match serde_json::to_value(__result) {
                    Ok(v) => v,
                    Err(e) => serde_json::json!({ "error": format!("serialization error: {}", e) }),
                }
            }
        }
    };

    let expanded = quote! {
        #[allow(non_camel_case_types)]
        pub struct #struct_ident;

        #[async_trait::async_trait]
        impl agentix::tool_trait::Tool for #struct_ident {
            fn raw_tools(&self) -> Vec<agentix::raw::shared::ToolDefinition> {
                vec![#raw_tools_body]
            }

            async fn call(&self, name: &str, args: serde_json::Value) -> serde_json::Value {
                match name {
                    #call_arm
                    _ => serde_json::json!({"error": format!("unknown tool: {}", name)}),
                }
            }
        }
    };

    expanded.into()
}

fn tool_from_impl(attr: TokenStream, item: TokenStream) -> TokenStream {
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
            if method.sig.asyncness.is_none() {
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
            });
        }
    }

    let raw_tools_body = tool_methods.iter().map(|m| {
        let tool_name = &m.tool_name;
        let description = &m.description;
        let prop_inserts = m.params.iter().map(|p| {
            let pname = &p.name;
            let pdesc = &p.desc;
            let ty = &p.ty;
            quote! {{
                let schema = <#ty as agentix::schemars::JsonSchema>::json_schema(&mut __gen);
                let mut prop = serde_json::to_value(schema).unwrap();
                if let Some(obj) = prop.as_object_mut() {
                    obj.insert("description".to_string(), serde_json::json!(#pdesc));
                }
                properties.insert(#pname.to_string(), prop);
            }}
        });
        let required: Vec<&str> = m
            .params
            .iter()
            .filter(|p| !p.optional)
            .map(|p| p.name.as_str())
            .collect();
        quote! {{
            let mut __gen = agentix::schemars::SchemaGenerator::default();
            let mut properties = serde_json::Map::new();
            #(#prop_inserts)*

            let required: Vec<&str> = vec![#(#required),*];
            let mut parameters = serde_json::json!({
                "type": "object",
                "properties": properties,
                "required": required,
            });

            // 同样为 impl 块内的参数注入可能生成的 definitions
            let defs = __gen.take_definitions();
            if !defs.is_empty() {
                parameters["$defs"] = serde_json::to_value(defs).unwrap();
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
    });

    let call_arms = tool_methods.iter().map(|m| {
        let tool_name = &m.tool_name;
        let body = &m.body;
        let arg_parses = m.params.iter().map(|p| {
            let pname = syn::Ident::new(&p.name, Span::call_site());
            let pname_str = &p.name;
            let ty = &p.ty;
            quote! {
                let #pname: #ty = match serde_json::from_value(
                    args.get(#pname_str).cloned().unwrap_or(serde_json::Value::Null)
                ) {
                    Ok(v) => v,
                    Err(e) => return serde_json::json!({
                        "error": format!("invalid argument '{}': {}", #pname_str, e)
                    }),
                };
            }
        });
        quote! {
            #tool_name => {
                #(#arg_parses)*
                let __result = (async move || { #body })().await;
                match serde_json::to_value(__result) {
                    Ok(v) => v,
                    Err(e) => serde_json::json!({ "error": format!("serialization error: {}", e) }),
                }
            }
        }
    });

    let self_ty = &item_impl.self_ty;

    let expanded = quote! {
        #[async_trait::async_trait]
        impl agentix::tool_trait::Tool for #self_ty {
            fn raw_tools(&self) -> Vec<agentix::raw::shared::ToolDefinition> {
                vec![#(#raw_tools_body),*]
            }

            async fn call(&self, name: &str, args: serde_json::Value) -> serde_json::Value {
                match name {
                    #(#call_arms)*
                    _ => serde_json::json!({"error": format!("unknown tool: {}", name)}),
                }
            }
        }
    };

    expanded.into()
}
