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

fn has_streaming_attr(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|a| a.path().is_ident("streaming"))
}

fn strip_streaming_attr(attrs: &mut Vec<syn::Attribute>) {
    attrs.retain(|a| !a.path().is_ident("streaming"));
}

struct ToolMethod {
    tool_name:   String,
    description: String,
    params:      Vec<ParamInfo>,
    body:        syn::Block,
    output:      syn::ReturnType,
    streaming:   bool,
}

struct ParamInfo {
    name:     String,
    ty:       Type,
    desc:     String,
    optional: bool,
}

// ── #[tool] ───────────────────────────────────────────────────────────────────

/// Annotate an `impl Tool for X` block (or a single fn) to generate the full
/// `Tool` trait implementation.
///
/// ## Normal (async) methods
/// ```ignore
/// #[tool]
/// impl agentix::Tool for MyTool {
///     /// Greet someone.
///     /// name: the person's name
///     async fn greet(&self, name: String) -> String {
///         format!("Hello, {name}!")
///     }
/// }
/// ```
///
/// ## Streaming methods — add `#[streaming]` on the method
///
/// The body must evaluate to a `Stream<Item = ToolOutput>`. Use `async_stream::stream!`
/// and `yield` inside it. The macro calls `.boxed()` on the returned stream.
///
/// ```ignore
/// #[tool]
/// impl agentix::Tool for Counter {
///     /// Count from 1 to n.
///     /// n: upper bound
///     #[streaming]
///     fn count_to(&self, n: u32) {
///         async_stream::stream! {
///             for i in 1..=n {
///                 yield agentix::ToolOutput::Progress(format!("{i}/{n}"));
///             }
///             yield agentix::ToolOutput::Result(vec![agentix::request::Content::text(format!("{{ \"final\": {n} }}" ))]);
///         }
///     }
/// }
/// ```
///
/// Both styles can be mixed in the same impl block.
#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    // Single-function form: `#[tool]` on a standalone fn.
    if let Ok(mut item_fn) = syn::parse::<ItemFn>(item.clone()) {
        let is_streaming = has_streaming_attr(&item_fn.attrs);
        strip_streaming_attr(&mut item_fn.attrs);
        return tool_from_fn(attr, item_fn, is_streaming);
    }
    tool_from_impl(attr, item)
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
            params.push(ParamInfo { name, ty, desc, optional });
        }
    }

    let method = ToolMethod {
        tool_name,
        description,
        params,
        body: *item_fn.block,
        output: item_fn.sig.output,
        streaming: is_streaming,
    };

    let raw_tools_body = generate_raw_tools(&method);
    let call_arm = generate_call_arm(&method);

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
                        let err = format!("{{\"error\":\"unknown tool: {}\"}}", name);
                        use agentix::futures::StreamExt;
                        agentix::futures::stream::iter(vec![agentix::tool_trait::ToolOutput::Result(vec![agentix::request::Content::text(err)])]).boxed()
                    }
                }
            }
        }

        impl<T: agentix::tool_trait::Tool + 'static> std::ops::Add<T> for #struct_ident {
            type Output = agentix::tool_trait::ToolBundle;
            fn add(self, rhs: T) -> Self::Output {
                agentix::tool_trait::ToolBundle::new() + self + rhs
            }
        }
    };

    expanded.into()
}

fn tool_from_impl(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut item_impl = parse_macro_input!(item as ItemImpl);

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

    for item in &mut item_impl.items {
        if let ImplItem::Fn(method) = item {
            let is_streaming = has_streaming_attr(&method.attrs);
            // Skip non-async, non-streaming methods.
            if !is_streaming && method.sig.asyncness.is_none() {
                continue;
            }
            strip_streaming_attr(&mut method.attrs);

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
                    params.push(ParamInfo { name, ty, desc, optional });
                }
            }
            tool_methods.push(ToolMethod {
                tool_name,
                description,
                params,
                body: method.block.clone(),
                output: method.sig.output.clone(),
                streaming: is_streaming,
            });
        }
    }

    let raw_tools_body = tool_methods.iter().map(generate_raw_tools);
    let call_arms = tool_methods.iter().map(generate_call_arm);
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
                        let err = format!("{{\"error\":\"unknown tool: {}\"}}", name);
                        use agentix::futures::StreamExt;
                        agentix::futures::stream::iter(vec![agentix::tool_trait::ToolOutput::Result(vec![agentix::request::Content::text(err)])]).boxed()
                    }
                }
            }
        }

        impl<T: agentix::tool_trait::Tool + 'static> std::ops::Add<T> for #self_ty {
            type Output = agentix::tool_trait::ToolBundle;
            fn add(self, rhs: T) -> Self::Output {
                agentix::tool_trait::ToolBundle::new() + self + rhs
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

fn generate_call_arm(method: &ToolMethod) -> proc_macro2::TokenStream {
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
                    let err = format!("{{\"error\":\"invalid argument '{}': {}\"}}", #pname_str, e);
                    use agentix::futures::StreamExt;
                    return agentix::futures::stream::iter(vec![agentix::tool_trait::ToolOutput::Result(vec![agentix::request::Content::text(err)])]).boxed();
                }
            };
        }
    });

    if method.streaming {
        // The method body must evaluate to a Stream. The user writes:
        //   async_stream::stream! { ... yield ToolOutput::... ... }
        // and we call .boxed() on the result.
        quote! {
            #tool_name => {
                use agentix::futures::StreamExt;
                #(#arg_parses)*
                let __stream = #body;
                __stream.boxed()
            }
        }
    } else {
        quote! {
            #tool_name => {
                use agentix::futures::StreamExt;
                #[allow(unused_imports)]
                use agentix::serde_json::Value;
                #(#arg_parses)*

                let __result = (async move || #output { #body })().await;

                #[allow(unused_imports)]
                use agentix::tool_trait::{ToolResultContent, ToolResultResult, ToolResultValue};

                let __val = (__result).__agentix_wrap();
                agentix::futures::stream::iter(vec![agentix::tool_trait::ToolOutput::Result(__val)]).boxed()
            }
        }
    }
}
