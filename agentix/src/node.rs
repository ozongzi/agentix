use futures::stream::BoxStream;
use crate::msg::AgentInput;
use crate::context::SharedContext;

// ── Node trait ────────────────────────────────────────────────────────────────

/// The fundamental unit of composition in a stream-based architecture.
///
/// A `Node` is a stream transformer: it takes an input stream and returns
/// an output stream. This allows for native Rust control flow (loops, branches)
/// by combining streams.
pub trait Node: Send + 'static {
    type Input:  Send + 'static;
    type Output: Send + 'static;

    fn run(self, input: BoxStream<'static, Self::Input>) -> BoxStream<'static, Self::Output>;
}

// ── Identity Node (Middleware Example) ────────────────────────────────────────

/// A node that passes messages through unchanged, allowing for side-effects.
pub struct TapNode<I: Send + 'static> {
    callback: Box<dyn Fn(&I) + Send + Sync + 'static>,
}

impl<I: Send + 'static> TapNode<I> {
    pub fn new(f: impl Fn(&I) + Send + Sync + 'static) -> Self {
        Self { callback: Box::new(f) }
    }
}

impl<I: Send + 'static> Node for TapNode<I> {
    type Input = I;
    type Output = I;

    fn run(self, input: BoxStream<'static, Self::Input>) -> BoxStream<'static, Self::Output> {
        use futures::StreamExt;
        let cb = self.callback;
        input.map(move |item| {
            (cb)(&item);
            item
        }).boxed()
    }
}

// ── Simple Prompt Template Node ───────────────────────────────────────────────

pub struct PromptNode {
    template: String,
    context:  Option<SharedContext>,
}

impl PromptNode {
    pub fn new(template: impl Into<String>) -> Self {
        Self { template: template.into(), context: None }
    }

    pub fn context(mut self, ctx: SharedContext) -> Self {
        self.context = Some(ctx);
        self
    }
}

impl Node for PromptNode {
    type Input = String;
    type Output = AgentInput;

    fn run(self, input: BoxStream<'static, Self::Input>) -> BoxStream<'static, Self::Output> {
        use futures::StreamExt;
        let tmpl = self.template;
        let ctx = self.context;
        
        input.map(move |text| {
            let mut rendered = tmpl.replace("{input}", &text);
            if let Some(ref c) = ctx {
                for (k, _) in c.snapshot() {
                    let ph = format!("{{{k}}}");
                    if rendered.contains(&ph) {
                        if let Some(val) = c.get_str(&k) {
                            rendered = rendered.replace(&ph, &val);
                        }
                    }
                }
            }
            AgentInput::User(vec![rendered.into()])
        }).boxed()
    }
}
