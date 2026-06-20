//! Search-results compressor — port of headroom's `search_compressor`.
//!
//! Compresses grep / ripgrep / ag output (one of the most common tool outputs
//! in coding tasks). Parses `file:line:content` (and ripgrep `-C` context with
//! mixed `:`/`-` separators), scores matches by error/warning signals and
//! optional context-word overlap, then keeps the most relevant per file under
//! an adaptive global cap. Typical compression: 5-10×.

use std::collections::{BTreeMap, BTreeSet};

use super::adaptive_sizer::compute_optimal_k;

#[derive(Debug, Clone, PartialEq)]
pub struct SearchMatch {
    pub file: String,
    pub line_number: u64,
    pub content: String,
    pub score: f32,
}

impl SearchMatch {
    fn new(file: impl Into<String>, line_number: u64, content: impl Into<String>) -> Self {
        Self {
            file: file.into(),
            line_number,
            content: content.into(),
            score: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FileMatches {
    pub file: String,
    pub matches: Vec<SearchMatch>,
}

impl FileMatches {
    fn new(file: impl Into<String>) -> Self {
        Self {
            file: file.into(),
            matches: Vec::new(),
        }
    }
    fn total_score(&self) -> f32 {
        self.matches.iter().map(|m| m.score).sum()
    }
}

#[derive(Debug, Clone)]
pub struct SearchCompressorConfig {
    pub max_matches_per_file: usize,
    pub always_keep_first: bool,
    pub always_keep_last: bool,
    pub max_total_matches: usize,
    pub max_files: usize,
    pub boost_errors: bool,
    pub min_matches_to_compress: usize,
}

impl Default for SearchCompressorConfig {
    fn default() -> Self {
        Self {
            max_matches_per_file: 5,
            always_keep_first: true,
            always_keep_last: true,
            max_total_matches: 30,
            max_files: 15,
            boost_errors: true,
            min_matches_to_compress: 10,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchCompressionResult {
    pub compressed: String,
    pub original_match_count: usize,
    pub compressed_match_count: usize,
    pub files_affected: usize,
    pub compression_ratio: f64,
}

pub struct SearchCompressor {
    config: SearchCompressorConfig,
}

impl SearchCompressor {
    pub fn new(config: SearchCompressorConfig) -> Self {
        Self { config }
    }

    pub fn compress(&self, content: &str, context: &str, bias: f64) -> SearchCompressionResult {
        let parsed = self.parse_search_results(content);
        let original_count: usize = parsed.values().map(|fm| fm.matches.len()).sum();

        if original_count < self.config.min_matches_to_compress {
            return SearchCompressionResult {
                compressed: content.to_string(),
                original_match_count: original_count,
                compressed_match_count: original_count,
                files_affected: parsed.len(),
                compression_ratio: 1.0,
            };
        }

        let mut scored = parsed;
        self.score_matches(&mut scored, context);
        let selected = self.select_matches(&scored, bias);
        let compressed = self.format_output(&selected, &scored);
        let compressed_count: usize = selected.values().map(|fm| fm.matches.len()).sum();
        let ratio = compressed.len() as f64 / content.len().max(1) as f64;

        SearchCompressionResult {
            compressed,
            original_match_count: original_count,
            compressed_match_count: compressed_count,
            files_affected: scored.len(),
            compression_ratio: ratio,
        }
    }

    fn parse_search_results(&self, content: &str) -> BTreeMap<String, FileMatches> {
        let mut out: BTreeMap<String, FileMatches> = BTreeMap::new();
        for raw in content.split('\n') {
            let line = raw.trim();
            if line.is_empty() {
                continue;
            }
            if let Some((file, line_no, body)) = parse_match_line(line) {
                out.entry(file.to_string())
                    .or_insert_with(|| FileMatches::new(file))
                    .matches
                    .push(SearchMatch::new(file, line_no, body));
            }
        }
        out
    }

    fn score_matches(&self, files: &mut BTreeMap<String, FileMatches>, context: &str) {
        let context_lower = context.to_ascii_lowercase();
        let context_words: Vec<&str> = context_lower
            .split_whitespace()
            .filter(|w| w.len() > 2)
            .collect();

        for fm in files.values_mut() {
            for m in &mut fm.matches {
                let mut score: f32 = 0.0;
                let content_lower = m.content.to_ascii_lowercase();

                for w in &context_words {
                    if content_lower.contains(w) {
                        score += 0.3;
                    }
                }

                if self.config.boost_errors {
                    score += importance_boost(&content_lower);
                }

                m.score = score.min(1.0);
            }
        }
    }

    fn select_matches(
        &self,
        files: &BTreeMap<String, FileMatches>,
        bias: f64,
    ) -> BTreeMap<String, FileMatches> {
        let mut by_score: Vec<(&String, &FileMatches)> = files.iter().collect();
        by_score.sort_by(|a, b| {
            b.1.total_score()
                .partial_cmp(&a.1.total_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if by_score.len() > self.config.max_files {
            by_score.truncate(self.config.max_files);
        }

        let all_match_strings: Vec<String> = by_score
            .iter()
            .flat_map(|(file, fm)| {
                fm.matches
                    .iter()
                    .map(move |m| format!("{}:{}:{}", file, m.line_number, m.content))
            })
            .collect();
        let all_refs: Vec<&str> = all_match_strings.iter().map(|s| s.as_str()).collect();
        let adaptive_total =
            compute_optimal_k(&all_refs, bias, 5, Some(self.config.max_total_matches));

        let mut selected: BTreeMap<String, FileMatches> = BTreeMap::new();
        let mut total_selected: usize = 0;

        for (file, fm) in by_score {
            if total_selected >= adaptive_total {
                continue;
            }

            let mut sorted = fm.matches.clone();
            sorted.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.line_number.cmp(&b.line_number))
            });

            let mut file_selected: Vec<SearchMatch> = Vec::new();
            let mut seen: BTreeSet<(u64, u64)> = BTreeSet::new();

            let remaining_cap = self
                .config
                .max_matches_per_file
                .min(adaptive_total.saturating_sub(total_selected));

            let mut push_unique = |m: &SearchMatch, file_selected: &mut Vec<SearchMatch>| {
                let key = (m.line_number, hash_u64(&m.content));
                if seen.insert(key) {
                    file_selected.push(m.clone());
                }
            };

            if self.config.always_keep_first {
                if let Some(first) = fm.matches.first() {
                    if file_selected.len() < remaining_cap {
                        push_unique(first, &mut file_selected);
                    }
                }
            }
            if self.config.always_keep_last && fm.matches.len() > 1 {
                if let Some(last) = fm.matches.last() {
                    if file_selected.len() < remaining_cap {
                        push_unique(last, &mut file_selected);
                    }
                }
            }
            for m in &sorted {
                if file_selected.len() >= remaining_cap {
                    break;
                }
                push_unique(m, &mut file_selected);
            }

            file_selected.sort_by_key(|m| m.line_number);
            total_selected += file_selected.len();
            selected.insert(
                file.clone(),
                FileMatches {
                    file: file.clone(),
                    matches: file_selected,
                },
            );
        }

        selected
    }

    fn format_output(
        &self,
        selected: &BTreeMap<String, FileMatches>,
        original: &BTreeMap<String, FileMatches>,
    ) -> String {
        let mut lines: Vec<String> = Vec::new();
        for (file, fm) in selected {
            for m in &fm.matches {
                lines.push(format!("{}:{}:{}", m.file, m.line_number, m.content));
            }
            if let Some(orig_fm) = original.get(file) {
                if orig_fm.matches.len() > fm.matches.len() {
                    let omitted = orig_fm.matches.len() - fm.matches.len();
                    lines.push(format!("[... and {} more matches in {}]", omitted, file));
                }
            }
        }
        lines.join("\n")
    }
}

/// Boost grep matches that look like errors/warnings/important markers.
fn importance_boost(content_lower: &str) -> f32 {
    const ERROR_KW: &[&str] = &["error", "panic", "exception", "fatal", "failed", "failure"];
    const WARN_KW: &[&str] = &["warning", "deprecated", "todo", "fixme"];
    const IMPORTANCE_KW: &[&str] = &["def ", "class ", "fn ", "function ", "impl ", "struct "];

    if ERROR_KW.iter().any(|k| content_lower.contains(k)) {
        0.5
    } else if WARN_KW.iter().any(|k| content_lower.contains(k)) {
        0.4
    } else if IMPORTANCE_KW.iter().any(|k| content_lower.contains(k)) {
        0.3
    } else {
        0.0
    }
}

/// Parse one grep/ripgrep-style line into `(file, line_number, content)`.
fn parse_match_line(line: &str) -> Option<(&str, u64, &str)> {
    let bytes = line.as_bytes();
    let scan_start = if bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && (bytes[2] == b'\\' || bytes[2] == b'/')
    {
        2
    } else {
        0
    };

    let mut i = scan_start;
    while i < bytes.len() {
        if bytes[i] == b':' || bytes[i] == b'-' {
            if i > 0 && (bytes[i - 1] == b':' || bytes[i - 1] == b'-') {
                i += 1;
                continue;
            }
            let digits_start = i + 1;
            let mut j = digits_start;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            if j > digits_start && j < bytes.len() && (bytes[j] == b':' || bytes[j] == b'-') {
                if i == 0 {
                    return None;
                }
                let (file, rest) = line.split_at(i);
                let line_no = std::str::from_utf8(&bytes[digits_start..j])
                    .ok()
                    .and_then(|s| s.parse::<u64>().ok())?;
                // `rest` starts at the first separator; content begins just
                // after the second separator at absolute index `j`.
                let content = rest.split_at(j - i + 1).1;
                return Some((file, line_no, content));
            }
        }
        i += 1;
    }
    None
}

fn hash_u64(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cmp() -> SearchCompressor {
        SearchCompressor::new(SearchCompressorConfig::default())
    }

    #[test]
    fn parses_standard_grep_line() {
        assert_eq!(
            parse_match_line("src/utils.py:42:def process(items):"),
            Some(("src/utils.py", 42, "def process(items):"))
        );
    }

    #[test]
    fn parses_ripgrep_context_line() {
        assert_eq!(
            parse_match_line("src/main.py-40-some context"),
            Some(("src/main.py", 40, "some context"))
        );
    }

    #[test]
    fn parses_windows_path() {
        assert_eq!(
            parse_match_line(r"C:\Users\foo\bar.py:42:line"),
            Some((r"C:\Users\foo\bar.py", 42, "line"))
        );
    }

    #[test]
    fn small_result_sets_pass_through() {
        let content = "a.py:1:x\nb.py:2:y";
        let r = cmp().compress(content, "", 1.0);
        assert_eq!(r.compressed, content);
        assert_eq!(r.compression_ratio, 1.0);
    }

    #[test]
    fn compresses_large_result_set_and_keeps_errors() {
        let mut lines: Vec<String> = Vec::new();
        for i in 0..50 {
            lines.push(format!("src/mod_{}.rs:{}:    let x = {};", i % 5, i, i));
        }
        lines.push("src/mod_0.rs:999:    panic!(\"fatal error here\");".to_string());
        let content = lines.join("\n");

        let r = cmp().compress(&content, "", 1.0);
        assert!(r.compressed_match_count < r.original_match_count);
        assert!(
            r.compressed.contains("fatal error here"),
            "error match must survive: {}",
            r.compressed
        );
        assert!(r.compressed.contains("more matches in"));
    }
}
