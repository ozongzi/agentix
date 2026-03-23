use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use futures::StreamExt;
use tokio::sync::mpsc;
use tokio::task::AbortHandle;

use crate::bus::EventBus;
use crate::context::SharedContext;
use crate::msg::Msg;
use crate::request::UserContent;

// ── Node trait ────────────────────────────────────────────────────────────────

/// The unit of composition in an agentix graph.
///
/// A `Node` has two channel endpoints:
/// - [`input`][Node::input] — push `Msg` values *into* the node.
/// - [`output`][Node::output] — an [`EventBus`] the node publishes *to*.
///   Use [`EventBus::subscribe`] for the raw streaming view, or
///   [`EventBus::subscribe_assembled`] for the folded (one-per-turn) view.
///
/// [`Agent`][crate::Agent] implements `Node` out of the box.
pub trait Node: Send + Sync + 'static {
    fn input(&self) -> mpsc::Sender<Msg>;
    fn output(&self) -> EventBus;
}

// ── Agent implements Node ─────────────────────────────────────────────────────

impl Node for crate::agent::Agent {
    fn input(&self) -> mpsc::Sender<Msg> { self.inbox_sender() }
    fn output(&self) -> EventBus          { self.event_bus() }
}

// ── Middleware ────────────────────────────────────────────────────────────────

/// A function that intercepts messages flowing through a [`Graph`] edge.
///
/// Return `Some(msg)` to forward (possibly transformed), `None` to drop.
pub type MiddlewareFn = Arc<dyn Fn(Msg) -> Option<Msg> + Send + Sync + 'static>;

fn apply_middlewares(middlewares: &[MiddlewareFn], msg: Msg) -> Option<Msg> {
    middlewares.iter().fold(Some(msg), |acc, mw| acc.and_then(|m| mw(m)))
}

// ── GraphHandle ───────────────────────────────────────────────────────────────

/// RAII guard for all background tasks spawned by a [`Graph`].
///
/// Obtained via [`Graph::into_handle`].  Dropping it aborts every edge task
/// immediately.  If you don't need early cancellation, you can skip
/// `into_handle()` entirely — tasks will terminate naturally once the upstream
/// node's [`EventBus`] is dropped (the broadcast channel closes).
///
/// ```no_run
/// # #[tokio::main] async fn main() {
/// let a = agentix::deepseek(std::env::var("KEY").unwrap());
/// let b = agentix::deepseek(std::env::var("KEY").unwrap());
///
/// let _handle = agentix::Graph::new()
///     .edge(&a, &b)
///     .into_handle();   // graph runs until _handle is dropped
/// # }
/// ```
pub struct GraphHandle(Vec<AbortHandle>);

impl Drop for GraphHandle {
    fn drop(&mut self) {
        self.0.iter().for_each(|h| h.abort());
    }
}

// ── Graph ─────────────────────────────────────────────────────────────────────

/// Declarative wiring of [`Node`]s connected by channels.
///
/// Every edge spawns a background task.  Call [`into_handle`][Graph::into_handle]
/// to obtain a [`GraphHandle`] that aborts those tasks when dropped.
///
/// # Edge semantics
///
/// `edge(&from, &to)` uses the **default LLM-chain transform**: assembled
/// `Token(text)` → `User(text)`, and `Custom(_)` passes through unchanged.
/// Use [`edge_map`][Graph::edge_map] for custom transforms.
///
/// # Example
/// ```no_run
/// # #[tokio::main] async fn main() {
/// let summariser = agentix::deepseek(std::env::var("KEY").unwrap())
///     .system_prompt("Summarise in one sentence.");
/// let translator = agentix::deepseek(std::env::var("KEY").unwrap())
///     .system_prompt("Translate to French.");
///
/// let _handle = agentix::Graph::new()
///     .middleware(|msg| { println!("edge: {msg:?}"); Some(msg) })
///     .edge(&summariser, &translator)
///     .into_handle();
///
/// summariser.send("Long article…").await;
/// # }
/// ```
pub struct Graph {
    middlewares: Vec<MiddlewareFn>,
    handles:     Vec<AbortHandle>,
}

impl Default for Graph {
    fn default() -> Self { Self { middlewares: vec![], handles: vec![] } }
}

impl Graph {
    pub fn new() -> Self { Self::default() }

    /// Register a middleware that runs on every message crossing any edge.
    /// Middlewares are applied in registration order; returning `None` drops
    /// the message.
    pub fn middleware(
        mut self,
        f: impl Fn(Msg) -> Option<Msg> + Send + Sync + 'static,
    ) -> Self {
        self.middlewares.push(Arc::new(f));
        self
    }

    /// Wire `from` → `to` using the default LLM-chain transform:
    /// - `Token(text)` → `User(text)` (assembled LLM output becomes next input)
    /// - `Custom(_)` passes through unchanged (typed payloads are preserved)
    /// - All other variants are dropped
    ///
    /// For a custom transform use [`edge_map`][Graph::edge_map].
    pub fn edge(self, from: &impl Node, to: &impl Node) -> Self {
        self.edge_map(from, to, |msg| match msg {
            Msg::Token(text) => Some(Msg::User(vec![UserContent::Text(text)])), // LLM output → next agent input
            Msg::User(_)     => Some(msg),             // PromptTemplate / OutputParser output
            Msg::Custom(_)   => Some(msg),             // typed payloads pass through
            _                => None,
        })
    }

    /// Wire `from` → `to` with a **caller-supplied message transform**.
    ///
    /// `map_fn` receives each assembled message from `from`'s output.  Return
    /// `Some(msg)` to forward (possibly transformed), or `None` to drop.
    /// Registered middlewares run *after* `map_fn`.
    ///
    /// This is the escape hatch for typed data: emit `Msg::Custom(payload)`
    /// from one node and forward it here without lossy string conversion.
    ///
    /// # Example
    /// ```no_run
    /// # use agentix::{Graph, Msg};
    /// # #[tokio::main] async fn main() {
    /// let extractor = agentix::deepseek(std::env::var("KEY").unwrap());
    /// let writer    = agentix::deepseek(std::env::var("KEY").unwrap());
    ///
    /// let _h = Graph::new().edge_map(&extractor, &writer, |msg| {
    ///     match msg {
    ///         Msg::Custom(_)   => Some(msg),          // pass typed payload
    ///         Msg::Token(t)    => Some(Msg::User(vec![t.into()])), // fallback for plain text
    ///         _                => None,
    ///     }
    /// }).into_handle();
    /// # }
    /// ```
    pub fn edge_map(
        mut self,
        from: &impl Node,
        to: &impl Node,
        map_fn: impl Fn(Msg) -> Option<Msg> + Send + Sync + 'static,
    ) -> Self {
        let bus        = from.output();
        let mut stream = Box::pin(bus.subscribe_assembled());
        let tx         = to.input();
        let mws        = self.middlewares.clone();

        let jh = tokio::spawn(async move {
            while let Some(msg) = stream.next().await {
                if let Some(mapped) = map_fn(msg) {
                    if let Some(out) = apply_middlewares(&mws, mapped) {
                        let _ = tx.send(out).await;
                    }
                }
            }
        });

        self.handles.push(jh.abort_handle());
        self
    }

    /// Number of edges wired so far.
    pub fn edge_count(&self) -> usize { self.handles.len() }

    /// Consume the graph and return a [`GraphHandle`].
    ///
    /// The handle **must be kept alive** for as long as the graph should run.
    /// Dropping it aborts all edge tasks.
    #[must_use]
    pub fn into_handle(self) -> GraphHandle {
        GraphHandle(self.handles)
    }
}

// ── PromptTemplate ────────────────────────────────────────────────────────────

/// A lightweight [`Node`] that renders a prompt template before forwarding.
///
/// When the node receives `Msg::User(text)`, substitutions are applied in order:
/// 1. `{input}` → the incoming text
/// 2. `{key}` → values registered with [`.var()`][PromptTemplate::var]
/// 3. `{key}` → values from the attached [`SharedContext`] (fallback)
///
/// All other `Msg` variants pass through unchanged.
///
/// # Example
/// ```no_run
/// # #[tokio::main] async fn main() {
/// use agentix::{Graph, Node, PromptTemplate, SharedContext};
///
/// let ctx = SharedContext::new();
/// ctx.set("lang", "French");
///
/// let prompt = PromptTemplate::new("Translate {input} to {lang}")
///     .context(ctx.clone());   // reads {lang} from the shared context
/// let agent  = agentix::deepseek(std::env::var("KEY").unwrap());
///
/// let _h = Graph::new().edge(&prompt, &agent).into_handle();
/// prompt.input().send(agentix::Msg::User(vec!["Hello world".into()])).await.unwrap();
///
/// // Hot-swap the language without rebuilding the graph:
/// ctx.set("lang", "Japanese");
/// # }
/// ```
pub struct PromptTemplate {
    inbox_tx: mpsc::Sender<Msg>,
    bus:      EventBus,
    vars:     Arc<RwLock<HashMap<String, String>>>,
    tmpl:     Arc<String>,
}

fn spawn_template_task(
    mut rx:  mpsc::Receiver<Msg>,
    bus:     EventBus,
    vars:    Arc<RwLock<HashMap<String, String>>>,
    tmpl:    Arc<String>,
    ctx:     Option<SharedContext>,
) {
    tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            let out = match msg {
                Msg::User(parts) => {
                    let text: String = parts.iter()
                        .filter_map(|p| if let UserContent::Text(t) = p { Some(t.as_str()) } else { None })
                        .collect::<Vec<_>>().join("\n");
                    let mut s = tmpl.replace("{input}", &text);
                    {
                        let v = vars.read().unwrap();
                        for (k, val) in v.iter() {
                            s = s.replace(&format!("{{{k}}}"), val);
                        }
                    }
                    if let Some(ref c) = ctx {
                        for (k, _) in c.snapshot() {
                            let ph = format!("{{{k}}}");
                            if s.contains(&ph) {
                                if let Some(val) = c.get_str(&k) {
                                    s = s.replace(&ph, &val);
                                }
                            }
                        }
                    }
                    Msg::User(vec![UserContent::Text(s)])
                }
                other => other,
            };
            bus.send(out);
        }
    });
}

impl PromptTemplate {
    /// Create a new template node.  May only be called inside a Tokio runtime.
    pub fn new(template: impl Into<String>) -> Self {
        let tmpl = Arc::new(template.into());
        let (tx, rx) = mpsc::channel::<Msg>(64);
        let bus      = EventBus::new(512);
        let vars: Arc<RwLock<HashMap<String, String>>> = Default::default();

        spawn_template_task(rx, bus.clone(), Arc::clone(&vars), Arc::clone(&tmpl), None);

        Self { inbox_tx: tx, bus, vars, tmpl }
    }

    /// Pre-set a named variable.  `.var()` takes precedence over [`SharedContext`].
    pub fn var(self, key: &str, value: impl Into<String>) -> Self {
        self.vars.write().unwrap().insert(key.to_string(), value.into());
        self
    }

    /// Attach a [`SharedContext`].  Placeholders not resolved by `.var()` are
    /// looked up in `ctx` at render time — enabling hot-swappable values.
    ///
    /// Calling `context()` replaces the background task so the new context is
    /// used for all subsequent messages.  The old task exits when its channel
    /// is closed.
    pub fn context(mut self, ctx: SharedContext) -> Self {
        let (tx, rx) = mpsc::channel::<Msg>(64);
        // Replacing inbox_tx drops the old sender; old task exits when it drains.
        self.inbox_tx = tx;
        spawn_template_task(rx, self.bus.clone(), Arc::clone(&self.vars), Arc::clone(&self.tmpl), Some(ctx));
        self
    }
}

impl Node for PromptTemplate {
    fn input(&self) -> mpsc::Sender<Msg> { self.inbox_tx.clone() }
    fn output(&self) -> EventBus          { self.bus.clone() }
}

// ── OutputParser ──────────────────────────────────────────────────────────────

/// A lightweight [`Node`] that transforms assembled text output.
///
/// When the node receives `Msg::User(text)` (which is what [`Graph::edge`]
/// delivers), it applies the parser function and re-emits the result as
/// `Msg::User(transformed)`.  All other variants pass through unchanged.
///
/// # Example
/// ```no_run
/// # async fn run() {
/// let agent  = agentix::deepseek(std::env::var("KEY").unwrap())
///     .system_prompt("Respond with only a JSON object: {\"score\": <0-10>}");
/// let parser = agentix::OutputParser::new(|s| {
///     // extract "score" field or default to "0"
///     serde_json::from_str::<serde_json::Value>(&s)
///         .ok()
///         .and_then(|v| v["score"].as_i64().map(|n| n.to_string()))
///         .unwrap_or_else(|| "0".into())
/// });
///
/// agentix::Graph::new().edge(&agent, &parser);
/// # }
/// ```
pub struct OutputParser {
    inbox_tx: mpsc::Sender<Msg>,
    bus:      EventBus,
}

impl OutputParser {
    /// Create a new output-parser node with a `String → String` transform.
    /// May only be called inside a Tokio runtime.
    pub fn new(f: impl Fn(String) -> String + Send + 'static) -> Self {
        let (tx, mut rx) = mpsc::channel::<Msg>(64);
        let bus          = EventBus::new(512);
        let bus_c        = bus.clone();

        tokio::spawn(async move {
            while let Some(msg) = rx.recv().await {
                let out = match msg {
                    Msg::User(parts) => {
                        let text: String = parts.into_iter()
                            .filter_map(|p| if let UserContent::Text(t) = p { Some(t) } else { None })
                            .collect::<Vec<_>>().join("\n");
                        Msg::User(vec![UserContent::Text(f(text))])
                    }
                    other           => other,
                };
                bus_c.send(out);
            }
        });

        Self { inbox_tx: tx, bus }
    }
}

impl Node for OutputParser {
    fn input(&self) -> mpsc::Sender<Msg> { self.inbox_tx.clone() }
    fn output(&self) -> EventBus          { self.bus.clone() }
}
