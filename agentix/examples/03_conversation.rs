use agentix::{ImageContent, ImageData, LlmEvent, Message, Request, UserContent};
use futures::StreamExt;
use std::env;
use std::io::{self, Write};

// A tiny base64 encoder to avoid adding external dependencies to the example.
fn encode_base64(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((data.len() / 3 + 1) * 4);
    for chunk in data.chunks(3) {
        let n = chunk.len();
        let n0 = chunk[0] as u32;
        let n1 = if n > 1 { chunk[1] as u32 } else { 0 };
        let n2 = if n > 2 { chunk[2] as u32 } else { 0 };
        let val = (n0 << 16) | (n1 << 8) | n2;
        out.push(CHARS[(val >> 18) as usize] as char);
        out.push(CHARS[((val >> 12) & 0x3F) as usize] as char);
        out.push(if n > 1 {
            CHARS[((val >> 6) & 0x3F) as usize] as char
        } else {
            '='
        });
        out.push(if n > 2 {
            CHARS[(val & 0x3F) as usize] as char
        } else {
            '='
        });
    }
    out
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // For this conversation example, we'll use Anthropic (Claude), which supports images.
    let api_key = env::var("ANTHROPIC_API_KEY")
        .expect("ANTHROPIC_API_KEY must be set in your environment variables");

    let http = reqwest::Client::new();

    // The system prompt is set on the Request itself,
    // so we only store the user/assistant messages in history.
    let mut messages: Vec<Message> = Vec::new();

    println!("Starting multimodal conversation with Claude. Type 'quit' or 'exit' to end.");
    println!("Hint: You can type `image:/path/to/pic.jpg` to send a local image.");

    loop {
        print!("\nUser: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") {
            break;
        }
        if input.is_empty() {
            continue;
        }

        let mut user_parts = Vec::new();

        // Check if the user wants to attach an image
        if let Some(path) = input.strip_prefix("image:") {
            let path = path.trim();
            match std::fs::read(path) {
                Ok(bytes) => {
                    let ext = path.split('.').last().unwrap_or("jpeg");
                    let mime_type = format!("image/{}", ext);

                    user_parts.push(UserContent::Image(ImageContent {
                        data: ImageData::Base64(encode_base64(&bytes)),
                        mime_type,
                    }));

                    // Add a default text prompt alongside the image
                    user_parts.push(UserContent::Text { text: "Please describe this image.".to_string() });
                    println!("(Attached local image: {path})");
                }
                Err(e) => {
                    eprintln!("Failed to read image at '{path}': {e}");
                    continue;
                }
            }
        } else {
            // Normal text message
            user_parts.push(UserContent::Text { text: input.to_string() });
        }

        // Add the user's message to the history
        messages.push(Message::User(user_parts));

        // Create the request using the full message history
        let mut stream = Request::anthropic(api_key.clone())
            .system_prompt(
                "You're a helpful assistant that can understand both text and images. Respond to the user's messages accordingly.",
            )
            .messages(messages.clone())
            .stream(&http)
            .await?;

        print!("Claude: ");
        io::stdout().flush()?;

        // Stream the response and collect the full text
        let mut assistant_reply = String::new();
        while let Some(event) = stream.next().await {
            match event {
                LlmEvent::Token(t) => {
                    print!("{t}");
                    io::stdout().flush()?;
                    assistant_reply.push_str(&t);
                }
                LlmEvent::Error(e) => {
                    eprintln!("\nError: {e}");
                }
                LlmEvent::Done => break,
                _ => {}
            }
        }
        println!();

        // Add the assistant's reply to the history so it has context for the next turn
        if !assistant_reply.is_empty() {
            messages.push(Message::Assistant {
                content: Some(assistant_reply),
                reasoning: None,
                tool_calls: vec![],
            });
        }
    }

    Ok(())
}
