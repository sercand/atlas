// SPDX-License-Identifier: AGPL-3.0-only

use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Response, Sse};
use futures::StreamExt;

use super::translator::*;

/// Encode the pipeline's neutral delta stream as Anthropic's structured
/// SSE event stream, **per-delta**: each `ir::StreamDelta` produces zero
/// or more Anthropic events emitted to the client before the next delta
/// is processed. (This replaces the old wrapper that re-parsed the
/// serialized OpenAI SSE bytes chunk-by-chunk — one fewer serialize/
/// parse round-trip and one fewer protocol to drift against.)
pub(super) fn anthropic_sse_from_deltas(
    deltas: crate::ir::DeltaStream,
    req_model: String,
) -> Response {
    // Match the chat_stream sizing — see chat_stream/mod.rs for rationale.
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(1024);

    tokio::spawn(async move {
        let mut translator = AnthropicTranslator::new(req_model);
        let mut pending: Vec<SseEvent> = Vec::new();
        let mut deltas = deltas;

        // Drain `pending` into the channel, converting each typed
        // `SseEvent` to an axum wire event. Returns `false` when the
        // receiver has hung up — caller should abort.
        async fn flush(
            tx: &tokio::sync::mpsc::Sender<Result<Event, std::convert::Infallible>>,
            pending: &mut Vec<SseEvent>,
        ) -> bool {
            for ev in pending.drain(..) {
                if tx.send(Ok(to_axum_event(&ev))).await.is_err() {
                    return false;
                }
            }
            true
        }

        while let Some(delta) = deltas.next().await {
            translator.on_delta(&delta, &mut pending);
            if !flush(&tx, &mut pending).await {
                return;
            }
        }

        // Ensure message_stop fires even if the stream ended without a
        // Finish delta.
        translator.finalize(&mut pending);
        if !flush(&tx, &mut pending).await {
            tracing::warn!("anthropic stream: final flush failed (receiver dropped)");
        }
    });

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}
