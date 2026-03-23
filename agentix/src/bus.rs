use std::future::Future;

use futures::Stream;
use tokio::sync::broadcast;

use crate::msg::Msg;

/// A cloneable broadcast bus for agent events.
///
/// All components publish to the same bus; external observers subscribe
/// independently.  Cloning is cheap — all clones share the same underlying
/// channel.
///
/// # Example
/// ```no_run
/// use agentix::EventBus;
///
/// let bus = EventBus::new(512);
///
/// // Attach a custom observer
/// bus.tap(|msg| async move {
///     println!("{msg:?}");
/// });
///
/// // Raw subscriber (e.g. for a WebSocket handler)
/// let mut rx = bus.subscribe();
/// ```
#[derive(Clone)]
pub struct EventBus {
    tx: broadcast::Sender<Msg>,
}

impl EventBus {
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self { tx }
    }

    /// Broadcast a message to all current subscribers.
    /// Silently drops the message if there are no subscribers.
    pub fn send(&self, msg: Msg) {
        let _ = self.tx.send(msg);
    }

    /// Subscribe to future messages.
    pub fn subscribe(&self) -> broadcast::Receiver<Msg> {
        self.tx.subscribe()
    }

    /// Subscribe to future messages, folding streaming fragments into complete events.
    ///
    /// Concretely:
    /// - Multiple [`Msg::Token`] chunks are buffered and emitted as a **single**
    ///   `Token(full_text)` just before the [`Msg::Done`] that ends the turn.
    /// - Multiple [`Msg::Reasoning`] chunks are folded the same way.
    /// - All other variants (`ToolCall`, `ToolResult`, `Done`, `User`, …) pass
    ///   through unchanged.
    ///
    /// This gives downstream nodes the same view a non-streaming provider would
    /// produce — same variant names, just assembled content.
    pub fn subscribe_assembled(&self) -> impl Stream<Item = Msg> + 'static {
        let mut rx = self.tx.subscribe();
        async_stream::stream! {
            let mut token_buf     = String::new();
            let mut reasoning_buf = String::new();
            loop {
                let msg = match rx.recv().await {
                    Ok(m)                                    => m,
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed)    => break,
                };
                match msg {
                    Msg::Token(t)    => token_buf.push_str(&t),
                    Msg::Reasoning(r) => reasoning_buf.push_str(&r),
                    Msg::Done => {
                        if !token_buf.is_empty() {
                            yield Msg::Token(std::mem::take(&mut token_buf));
                        }
                        if !reasoning_buf.is_empty() {
                            yield Msg::Reasoning(std::mem::take(&mut reasoning_buf));
                        }
                        yield Msg::Done;
                    }
                    other => yield other,
                }
            }
        }
    }

    /// Spawn a background task that calls `f` for every bus message.
    ///
    /// The task runs until the bus is dropped or the receiver lags too far
    /// behind (lagged messages are silently skipped).
    pub fn tap<F, Fut>(&self, f: F)
    where
        F:   Fn(Msg) -> Fut + Send + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let mut rx = self.tx.subscribe();
        tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(msg) => f(msg).await,
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        });
    }


}
