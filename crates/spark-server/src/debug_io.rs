// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_DEBUG_IO=1` — dump the two strings that bracket a generation:
//! the PROMPT the chat template rendered, and the RAW model output, logged
//! before anything parses, splits, or strips it.
//!
//! Why this exists. A tool-call bug is always one of three things: the
//! template rendered the wrong prompt, the model emitted the wrong text, or
//! the parser misread text that was fine. Without both endpoints you cannot
//! tell them apart, and the 2026-08-29 session spent an evening on exactly
//! that — a `</parameter=path>` close leaked into a parameter value, and the
//! only evidence was the already-parsed JSON, which cannot show what the
//! model actually wrote.
//!
//! The bug class this serves is INVISIBLE CHARACTERS: a NON-BREAKING HYPHEN
//! in `utf‑8`, a NO-BREAK SPACE inside `'Times New Roman'`, a zero-width
//! space anywhere. Logging the string verbatim is useless for those — they
//! render identically to the ASCII they replace. So every dump is followed
//! by a codepoint census naming each non-ASCII character and its offset.
//!
//! Off by default and costs one env read per call when off.

use std::fmt::Write as _;
use std::sync::OnceLock;

/// `ATLAS_DEBUG_IO=1` (presence of `1`, not mere presence — `=0` is off, so
/// it can be pinned off in a unit file that inherits a debug environment).
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_DEBUG_IO").as_deref() == Ok("1"))
}

/// Longest prefix dumped per string. A 160k-token prompt in the journal
/// helps nobody; the head is where the template's system block and tool
/// schemas live, and `ATLAS_DEBUG_IO_MAX` raises it when the tail is what
/// matters.
fn max_chars() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("ATLAS_DEBUG_IO_MAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16384)
    })
}

/// Every non-ASCII codepoint in `s`, as `U+XXXX NAME @offset` lines.
///
/// This is the part that earns the module. Two strings that print
/// identically to a terminal — and to a model rereading its own output —
/// differ here and nowhere else.
pub(crate) fn codepoint_census(s: &str) -> String {
    let mut seen: Vec<(char, usize, usize)> = Vec::new();
    for (off, ch) in s.char_indices() {
        if ch.is_ascii() {
            continue;
        }
        match seen.iter_mut().find(|(c, _, _)| *c == ch) {
            Some((_, _, n)) => *n += 1,
            None => seen.push((ch, off, 1)),
        }
    }
    if seen.is_empty() {
        return "  (pure ASCII)".to_string();
    }
    let mut out = String::new();
    for (ch, first, n) in seen {
        let _ = writeln!(
            out,
            "  U+{:04X} {:<34} x{n:<4} first@{first}",
            ch as u32,
            char_name(ch),
        );
    }
    out.pop();
    out
}

/// Names for the characters this bug class actually involves; everything
/// else reports its category, which is enough to spot "why is there a
/// format character in my HTML".
fn char_name(ch: char) -> &'static str {
    match ch {
        '\u{2010}' => "HYPHEN",
        '\u{2011}' => "NON-BREAKING HYPHEN",
        '\u{2012}' => "FIGURE DASH",
        '\u{2013}' => "EN DASH",
        '\u{2014}' => "EM DASH",
        '\u{2212}' => "MINUS SIGN",
        '\u{00AD}' => "SOFT HYPHEN (invisible)",
        '\u{2018}' => "LEFT SINGLE QUOTE",
        '\u{2019}' => "RIGHT SINGLE QUOTE",
        '\u{201C}' => "LEFT DOUBLE QUOTE",
        '\u{201D}' => "RIGHT DOUBLE QUOTE",
        '\u{00A0}' => "NO-BREAK SPACE",
        '\u{202F}' => "NARROW NO-BREAK SPACE",
        '\u{2009}' => "THIN SPACE",
        '\u{200B}' => "ZERO WIDTH SPACE (invisible)",
        '\u{200C}' => "ZERO WIDTH NON-JOINER (invisible)",
        '\u{200D}' => "ZERO WIDTH JOINER (invisible)",
        '\u{FEFF}' => "ZERO WIDTH NO-BREAK SPACE / BOM (invisible)",
        '\u{2026}' => "HORIZONTAL ELLIPSIS",
        c if c.is_whitespace() => "<whitespace>",
        c if (c as u32) < 0x2000 => "<letter or symbol>",
        _ => "<symbol>",
    }
}

fn head(s: &str) -> (&str, bool) {
    match s.char_indices().nth(max_chars()) {
        Some((cut, _)) => (&s[..cut], true),
        None => (s, false),
    }
}

/// The block both dumps emit. Pure so the exact bytes an operator will read
/// are asserted in tests, rather than only the census that feeds them.
pub(crate) fn render_dump(kind: &str, what: &str, text: &str) -> String {
    let (shown, truncated) = head(text);
    format!(
        "=== {kind} ({what}) === {} chars, {} bytes{}\n{}\n--- non-ASCII ---\n{}",
        text.chars().count(),
        text.len(),
        if truncated {
            " (TRUNCATED, raise ATLAS_DEBUG_IO_MAX)"
        } else {
            ""
        },
        shown,
        codepoint_census(text),
    )
}

/// The exact prompt bytes the chat template produced. Called from
/// `tokenizer::chat_render::render_chat`, the single place production
/// prompt bytes are made.
pub(crate) fn dump_prompt(prompt: &str) {
    if !enabled() {
        return;
    }
    tracing::info!(
        target: "atlas::debug_io",
        "{}",
        render_dump("TEMPLATE OUTPUT", "prompt", prompt)
    );
}

/// The raw model output, before tool parsing / think splitting / content
/// stripping. `what` names the call site so two dumps in one request are
/// distinguishable.
pub(crate) fn dump_output(what: &str, output: &str) {
    if !enabled() {
        return;
    }
    tracing::info!(
        target: "atlas::debug_io",
        "{}",
        render_dump("MODEL OUTPUT", what, output)
    );
}

/// The generated token ids, space separated. A substituted token shows up
/// here as one changed id, which separates "the model sampled the wrong
/// token" from "the detokenizer rendered the right token wrongly" — the
/// decoded text alone cannot.
pub(crate) fn dump_token_ids(ids: &[u32]) {
    if !enabled() || ids.is_empty() {
        return;
    }
    let shown = ids.len().min(max_chars() / 7);
    let mut s = String::with_capacity(shown * 7);
    for (i, id) in ids.iter().take(shown).enumerate() {
        if i > 0 {
            s.push(' ');
        }
        let _ = write!(s, "{id}");
    }
    tracing::info!(
        target: "atlas::debug_io",
        "=== MODEL OUTPUT (stream, token ids) === {} ids{}\n{}",
        ids.len(),
        if shown < ids.len() { " (TRUNCATED)" } else { "" },
        s,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn census_is_empty_for_ascii() {
        assert_eq!(codepoint_census("<!DOCTYPE html>"), "  (pure ASCII)");
    }

    /// The whole point: `utf-8` and `utf‑8` print the same and must not
    /// report the same. U+2011 is the character the 2026-08-29 screenshot
    /// could not distinguish by eye.
    #[test]
    fn census_names_the_lookalike_hyphen() {
        let ascii = codepoint_census("charset=\"utf-8\"");
        let sneaky = codepoint_census("charset=\"utf\u{2011}8\"");
        assert_eq!(ascii, "  (pure ASCII)");
        assert!(sneaky.contains("U+2011"), "{sneaky}");
        assert!(sneaky.contains("NON-BREAKING HYPHEN"), "{sneaky}");
        assert_ne!(ascii, sneaky);
    }

    /// Invisible characters are the ones a verbatim log cannot show.
    #[test]
    fn census_reports_invisibles_and_counts_repeats() {
        let c = codepoint_census("a\u{200B}b\u{00A0}c\u{00A0}d");
        assert!(c.contains("ZERO WIDTH SPACE"), "{c}");
        assert!(c.contains("NO-BREAK SPACE"), "{c}");
        assert!(c.contains("x2"), "repeat count missing: {c}");
    }


    /// The operator-facing block: header counts, the text itself, and the
    /// census — in that order, so a journal grep for the marker lands on
    /// everything needed to compare two runs.
    #[test]
    fn dump_block_carries_counts_text_and_census() {
        let d = render_dump("MODEL OUTPUT", "choice 0, content", "<function=x>caf\u{00E9}");
        assert!(d.starts_with("=== MODEL OUTPUT (choice 0, content) ==="), "{d}");
        assert!(d.contains("16 chars, 17 bytes"), "{d}");
        assert!(d.contains("<function=x>caf\u{00E9}"), "{d}");
        assert!(d.contains("U+00E9"), "{d}");
    }

    /// Byte count and char count must differ for non-ASCII — that gap is
    /// often the first hint that something re-encoded the payload.
    #[test]
    fn dump_reports_chars_and_bytes_separately() {
        let d = render_dump("TEMPLATE OUTPUT", "prompt", "\u{1F44B}");
        assert!(d.contains("1 chars, 4 bytes"), "{d}");
    }

    /// The two hooks must be distinguishable in one journal: a request logs
    /// the prompt and then the output, and an operator diffing them needs
    /// the markers to differ.
    #[test]
    fn prompt_and_output_markers_are_distinct() {
        let p = render_dump("TEMPLATE OUTPUT", "prompt", "x");
        let o = render_dump("MODEL OUTPUT", "choice 0, content", "x");
        assert!(p.contains("TEMPLATE OUTPUT") && !p.contains("MODEL OUTPUT"));
        assert!(o.contains("MODEL OUTPUT") && !o.contains("TEMPLATE OUTPUT"));
    }

    /// A tool call whose close leaked (the Fix A bug) is visible in the dump
    /// as raw text — this is the evidence the parsed JSON could not give.
    #[test]
    fn dump_shows_a_named_close_verbatim() {
        let raw = "<function=edit_file>\n<parameter=content>hi</parameter=content>\n</function>";
        let d = render_dump("MODEL OUTPUT", "choice 0, content", raw);
        assert!(d.contains("</parameter=content>"), "{d}");
    }

    #[test]
    fn census_records_first_offset_in_bytes() {
        // Byte offsets, not char indices: 'a'@0, 'é'@1 (2 bytes), 'b'@3,
        // '€'@4. Reporting char indices here would send a reader to the
        // wrong place in a hexdump, which is the only tool that settles
        // these arguments.
        let c = codepoint_census("a\u{00E9}b\u{20AC}");
        assert!(c.contains("first@1"), "{c}");
        assert!(c.contains("first@4"), "{c}");
    }
}
