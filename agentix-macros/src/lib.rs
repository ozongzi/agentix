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
    let mut params = std::collections::HashMap::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        if let Some((key, val)) = line.split_once(':') {
            let key = key.trim().to_string();
            let val = val.trim().to_string();
            if key.chars().all(|c| c.is_alphanumeric() || c == '_') && !val.is_empty() {
                params.insert(key, val);
                continue;
            }
        }
        if params.is_empty() {
            desc_lines.push(line.clone());
        }
    }
    (desc_lines.join(" ").trim().to_string(), params)
}

/// Recursively map a `syn::Type` to a JSON Schema snippet.
///
/// Matches on the *structure* of the type rather than its string representation,
/// so path aliases (`std::string::String`), references (`&str`), and generic
/// wrappers (`Option<T>`, `Vec<T>`) all resolve correctly.
fn type_to_json_schema(ty: &Type) -> TokenStream2 {
    match ty {
        // &str, &String, &T — strip the reference and recurse
        Type::Reference(r) => type_to_json_schema(&r.elem),

        Type::Path(tp) => {
            // Only look at the final path segment so that
            // `std::string::String` and `String` both work.
            let seg = match tp.path.segments.last() {
                Some(s) => s,
                None => return unsupported(ty),
            };

            match seg.ident.to_string().as_str() {
                "String" | "str" => quote!(serde_json::json!({"type": "string"})),
                "bool" => quote!(serde_json::json!({"type": "boolean"})),
                "f32" | "f64" => quote!(serde_json::json!({"type": "number"})),
                "u8" | "u16" | "u32" | "u64" | "u128" | "usize" | "i8" | "i16" | "i32" | "i64"
                | "i128" | "isize" => {
                    quote!(serde_json::json!({"type": "integer"}))
                }
                // Option<T> — recurse into T
                "Option" => match inner_type_arg(seg) {
                    Some(inner) => type_to_json_schema(inner),
                    None => unsupported(ty),
                },
                // Vec<T> — recurse into T for the items schema
                "Vec" => match inner_type_arg(seg) {
                    Some(inner) => {
                        let items = type_to_json_schema(inner);
                        quote!(serde_json::json!({"type": "array", "items": #items}))
                    }
                    None => unsupported(ty),
                },
                _ => unsupported(ty),
            }
        }

        _ => unsupported(ty),
    }
}

/// Emit a compile-time error pointing at the offending type.
fn unsupported(ty: &Type) -> TokenStream2 {
    syn::Error::new_spanned(
        ty,
        "unsupported type in #[tool]: use String, bool, f32/f64, \
         an integer primitive, Vec<T>, or Option<T>",
    )
    .to_compile_error()
}

/// Extract the first generic type argument from a path segment, e.g. the `T`
/// in `Option<T>` or `Vec<T>`.
fn inner_type_arg(seg: &syn::PathSegment) -> Option<&Type> {
    if let syn::PathArguments::AngleBracketed(args) = &seg.arguments
        && let Some(syn::GenericArgument::Type(ty)) = args.args.first()
    {
        return Some(ty);
    }
    None
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
            let schema = type_to_json_schema(&p.ty);
            quote! {{
                let mut prop = #schema;
                prop["description"] = serde_json::json!(#pdesc);
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
            let mut properties = serde_json::Map::new();
            #(#prop_inserts)*
            let required: Vec<&str> = vec![#(#required),*];
            agentix::raw::shared::ToolDefinition {
                kind: agentix::raw::shared::ToolKind::Function,
                function: agentix::raw::shared::FunctionDefinition {
                    name: #tool_name.to_string(),
                    description: Some(#description.to_string()),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    }),
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
            let schema = type_to_json_schema(&p.ty);
            quote! {{
                let mut prop = #schema;
                prop["description"] = serde_json::json!(#pdesc);
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
            let mut properties = serde_json::Map::new();
            #(#prop_inserts)*
            let required: Vec<&str> = vec![#(#required),*];
            agentix::raw::shared::ToolDefinition {
                kind: agentix::raw::shared::ToolKind::Function,
                function: agentix::raw::shared::FunctionDefinition {
                    name: #tool_name.to_string(),
                    description: Some(#description.to_string()),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    }),
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
