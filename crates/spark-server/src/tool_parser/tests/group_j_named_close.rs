// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::super::*;

// ────────────────────────────────────────────────────────────────────────
// Named parameter close (2026-08-29). The qwen3_coder OPENING tag carries
// the parameter name, so models mirror it onto the close and emit
// `</parameter=content>` instead of `</parameter>`.
//
// That string is invisible to all three of the parser's finders:
//   `</parameter>`  — the `=` breaks the match
//   `<parameter=`   — the `/` breaks the match
//   `</function>`   — unrelated
// so the value ran on PAST its own terminator, swallowed the next
// parameter's tag and text, and left that parameter EMPTY.
//
// Caught live against qwen3.8-flash-next while reproducing a tool round-trip
// (bench/qwen4_exp/tool_roundtrip.py): an `edit_file` call came back with
//   path:    ""
//   content: "</parameter=path>\n/tmp/probe_one.txt"
// A required argument silently going empty is the worst shape of this bug,
// because the tool still runs — it just runs on the wrong file.
// ────────────────────────────────────────────────────────────────────────

fn args_of(text: &str) -> serde_json::Map<String, serde_json::Value> {
    let (_, calls) = parse_tool_calls(text);
    assert_eq!(calls.len(), 1, "expected exactly one call from: {text:?}");
    serde_json::from_str(&calls[0].function.arguments).expect("arguments are JSON")
}

/// The exact live failure, reduced.
#[test]
fn named_close_does_not_swallow_the_next_parameter() {
    let a = args_of(
        "<function=edit_file>\n\
         <parameter=content>hello world</parameter=content>\n\
         <parameter=path>/tmp/probe_one.txt</parameter=path>\n\
         </function>",
    );
    assert_eq!(a["content"], "hello world");
    assert_eq!(
        a["path"], "/tmp/probe_one.txt",
        "the named close must terminate `content`, not leak into it"
    );
    assert!(
        !a["content"].as_str().unwrap().contains("</parameter"),
        "a closing tag leaked into the value: {:?}",
        a["content"]
    );
}

/// Mixed spellings in one call — models are not consistent within a block.
#[test]
fn bare_and_named_closes_mix_within_one_call() {
    let a = args_of(
        "<function=edit_file>\n\
         <parameter=path>/tmp/a.txt</parameter>\n\
         <parameter=content>body</parameter=content>\n\
         </function>",
    );
    assert_eq!(a["path"], "/tmp/a.txt");
    assert_eq!(a["content"], "body");
}

/// The named close must not fire on a value that legitimately CONTAINS the
/// text — only the first real terminator ends the value, and here the bare
/// close comes first.
#[test]
fn bare_close_still_wins_when_it_comes_first() {
    let a = args_of(
        "<function=write>\n\
         <parameter=content>see </parameter> in the docs</parameter>\n\
         </function>",
    );
    assert_eq!(a["content"], "see");
}

/// A value whose text mentions a named close AFTER a real bare close must
/// keep the bare close as the boundary.
#[test]
fn named_close_later_in_the_value_does_not_preempt_a_bare_close() {
    let a = args_of(
        "<function=write>\n\
         <parameter=content>real body</parameter>\n\
         <parameter=path>/tmp/b</parameter=path>\n\
         </function>",
    );
    assert_eq!(a["content"], "real body");
    assert_eq!(a["path"], "/tmp/b");
}

/// Truncation: `</parameter=` with no `>` before end-of-line is NOT a close.
/// It must fall through to the existing recovery branches rather than
/// consume the remainder of the buffer as a terminator.
#[test]
fn unterminated_named_close_falls_through_to_recovery() {
    let a = args_of(
        "<function=edit_file>\n\
         <parameter=content>body\n\
         <parameter=path>/tmp/c.txt</parameter>\n\
         </function>",
    );
    // `content` recovers at the next `<parameter=`; `path` must still land.
    assert_eq!(a["path"], "/tmp/c.txt");
    assert!(a.contains_key("content"));
}

/// Whitespace inside the named close, which models also produce.
#[test]
fn named_close_tolerates_inner_spacing() {
    let a = args_of(
        "<function=edit_file>\n\
         <parameter=content>x</parameter=content >\n\
         <parameter=path>/tmp/d</parameter>\n\
         </function>",
    );
    assert_eq!(a["content"], "x");
    assert_eq!(a["path"], "/tmp/d");
}

/// The pre-existing garbled close-reopen recovery (`</parameter<parameter=`,
/// P0-2 2026-07-09) must be untouched by the named-close branch.
#[test]
fn garbled_close_reopen_recovery_still_works() {
    let a = args_of(
        "<function=write>\n\
         <parameter=filePath>/tmp/e</parameter<parameter=content>body</parameter>\n\
         </function>",
    );
    assert_eq!(a["filePath"], "/tmp/e");
    assert_eq!(a["content"], "body");
}

/// Multi-line content with a named close — the realistic `edit_file` shape,
/// and the one that carries the non-ASCII this whole investigation started
/// from. The value must survive byte-exact, U+2011 included.
#[test]
fn named_close_preserves_multiline_and_non_ascii_content() {
    let body = "<!DOCTYPE html>\n<meta charset=\"utf\u{2011}8\">\n<p>caf\u{00E9} \u{2014} 30\u{00A0}days</p>";
    let text = format!(
        "<function=edit_file>\n\
         <parameter=content>{body}</parameter=content>\n\
         <parameter=path>/tmp/f.html</parameter=path>\n\
         </function>"
    );
    let a = args_of(&text);
    assert_eq!(a["content"], body, "content must round-trip byte-exact");
    assert_eq!(a["path"], "/tmp/f.html");
}

// ── Shapes captured live on 2026-08-29/30 ───────────────────────────────
// Both are the SAME failure: the close tag's final `>` token replaced by a
// look-alike. The token ids from ATLAS_DEBUG_IO named them exactly —
// `29` is `>`, and the stream carried `835` (`Ġ>` = " >") where `29` belonged.

/// `</parameter >` — token 835 (`Ġ>`) instead of 29 (`>`). Captured in the
/// streamed token ids of a live agentic turn.
#[test]
fn space_before_the_bracket_is_still_a_close() {
    let a = args_of(
        "<function=edit_file>\n\
         <parameter=content>BODY</parameter >\n\
         <parameter=path>/tmp/a.txt</parameter>\n\
         </function>",
    );
    assert_eq!(a["content"], "BODY");
    assert_eq!(a["path"], "/tmp/a.txt");
}

/// Named close with a trailing space, the two variants combined.
#[test]
fn named_close_with_trailing_space() {
    let a = args_of(
        "<function=edit_file>\n\
         <parameter=content>BODY</parameter=content >\n\
         <parameter=path>/tmp/b.txt</parameter>\n\
         </function>",
    );
    assert_eq!(a["content"], "BODY");
    assert_eq!(a["path"], "/tmp/b.txt");
}

/// `</parameter→` — U+2192 where the `>` token belonged. Not a well-formed
/// close (no `>` at all), so it goes down the recovery path; the orphan
/// strip must remove it rather than leave it as the value's tail.
#[test]
fn arrow_instead_of_the_bracket_is_stripped_not_kept() {
    let a = args_of(
        "<function=edit_file>\n\
         <parameter=old_text>\n</parameter\u{2192}\n\
         <parameter=path>email.html</parameter>\n\
         </function>",
    );
    assert_eq!(
        a["old_text"], "",
        "the mis-spelled close must not become the value: {:?}",
        a["old_text"]
    );
    assert_eq!(a["path"], "email.html");
}

/// The strip is bounded: a value that genuinely ends with the literal text
/// plus real content keeps it, because that is content, not a close.
#[test]
fn a_long_tail_after_the_tag_is_content_not_a_close() {
    let a = args_of(
        "<function=write>\n\
         <parameter=content>docs say </parameter is the closer\n\
         <parameter=path>/tmp/c</parameter>\n\
         </function>",
    );
    assert!(
        a["content"].as_str().unwrap().contains("is the closer"),
        "content was over-trimmed: {:?}",
        a["content"]
    );
    assert_eq!(a["path"], "/tmp/c");
}
