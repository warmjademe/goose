//! Parity test for the complexity model. Loads `~/.goose/complexity_router/`
//! and `parity_fixture.jsonl` produced by `dump_parity_fixture.py`, then
//! asserts the Rust pipeline matches Python within tolerance on every row.
//!
//! Skipped (not failed) if the bundle is absent — CI without local weights
//! just won't exercise this.

use std::path::PathBuf;

use goose::agents::complexity_router::ComplexityModel;
use serde::Deserialize;

const TOLERANCE: f32 = 1e-3;

#[derive(Debug, Deserialize)]
struct ParityRow {
    text: String,
    #[serde(default)]
    lang: Option<String>,
    expected_complexity: f32,
    expected_tool_calls_norm: f32,
}

fn bundle_dir() -> PathBuf {
    dirs::home_dir()
        .expect("home dir")
        .join(".goose")
        .join("complexity_router")
}

#[test]
fn parity_against_python_reference() {
    let dir = bundle_dir();
    let cfg_path = dir.join("config.json");
    let fixture_path = dir.join("parity_fixture.jsonl");
    if !cfg_path.exists() || !fixture_path.exists() {
        eprintln!(
            "skipping: missing {} or {}",
            cfg_path.display(),
            fixture_path.display()
        );
        return;
    }

    let model = ComplexityModel::load_from_dir(&dir).expect("load complexity model");

    let fixture_text = std::fs::read_to_string(&fixture_path).expect("read fixture");
    let rows: Vec<ParityRow> = fixture_text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("parse fixture row"))
        .collect();

    assert!(!rows.is_empty(), "fixture has no rows");

    let mut failures = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        let out = model.score(&row.text).expect("score");
        let dc = (out.complexity - row.expected_complexity).abs();
        let dt = (out.tool_calls_norm - row.expected_tool_calls_norm).abs();
        let preview: String = row.text.chars().take(60).collect();
        eprintln!(
            "[{:>2}] lang={:<8} expected=({:.4}, {:.4}) got=({:.4}, {:.4}) Δ=({:.4}, {:.4}) {:?}",
            i,
            row.lang.as_deref().unwrap_or("?"),
            row.expected_complexity,
            row.expected_tool_calls_norm,
            out.complexity,
            out.tool_calls_norm,
            dc,
            dt,
            preview,
        );
        if dc > TOLERANCE || dt > TOLERANCE {
            failures.push((i, dc, dt));
        }
    }

    if !failures.is_empty() {
        panic!(
            "{}/{} rows exceeded tolerance {} — see stderr",
            failures.len(),
            rows.len(),
            TOLERANCE,
        );
    }
}
