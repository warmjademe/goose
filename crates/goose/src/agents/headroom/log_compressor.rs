//! Log / build-output compressor — port of headroom's `log_compressor`.
//!
//! Compresses build and test output (pytest, npm, cargo, jest, make, generic).
//! Typical input: thousands of lines with a handful of real errors. Typical
//! compression: 10-50×. The pipeline classifies each line (level, stack-trace
//! membership, summary membership), scores it, then selects errors/fails/
//! warnings/stack-traces/summaries plus context, capped by an adaptive budget.

use std::collections::BTreeSet;
use std::sync::OnceLock;

use regex::Regex;

use super::adaptive_sizer::compute_optimal_k;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LogFormat {
    Pytest,
    Npm,
    Cargo,
    Jest,
    Make,
    Generic,
}

impl LogFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            LogFormat::Pytest => "pytest",
            LogFormat::Npm => "npm",
            LogFormat::Cargo => "cargo",
            LogFormat::Jest => "jest",
            LogFormat::Make => "make",
            LogFormat::Generic => "generic",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LogLevel {
    Error,
    Fail,
    Warn,
    Info,
    Debug,
    Trace,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct LogLine {
    pub line_number: usize,
    pub content: String,
    pub level: LogLevel,
    pub is_stack_trace: bool,
    pub is_summary: bool,
    pub score: f32,
}

impl PartialEq for LogLine {
    fn eq(&self, other: &Self) -> bool {
        self.line_number == other.line_number
    }
}
impl Eq for LogLine {}
impl PartialOrd for LogLine {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for LogLine {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.line_number.cmp(&other.line_number)
    }
}

impl LogLine {
    fn new(line_number: usize, content: impl Into<String>) -> Self {
        Self {
            line_number,
            content: content.into(),
            level: LogLevel::Unknown,
            is_stack_trace: false,
            is_summary: false,
            score: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LogCompressorConfig {
    pub max_errors: usize,
    pub error_context_lines: usize,
    pub keep_first_error: bool,
    pub keep_last_error: bool,
    pub max_stack_traces: usize,
    pub stack_trace_max_lines: usize,
    pub max_warnings: usize,
    pub dedupe_warnings: bool,
    pub keep_summary_lines: bool,
    pub max_total_lines: usize,
    pub min_lines_to_compress: usize,
}

impl Default for LogCompressorConfig {
    fn default() -> Self {
        Self {
            max_errors: 10,
            error_context_lines: 3,
            keep_first_error: true,
            keep_last_error: true,
            max_stack_traces: 3,
            stack_trace_max_lines: 20,
            max_warnings: 5,
            dedupe_warnings: true,
            keep_summary_lines: true,
            max_total_lines: 100,
            min_lines_to_compress: 50,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LogCompressionResult {
    pub compressed: String,
    pub original_line_count: usize,
    pub compressed_line_count: usize,
    pub format_detected: LogFormat,
    pub compression_ratio: f64,
}

pub struct LogCompressor {
    config: LogCompressorConfig,
}

impl LogCompressor {
    pub fn new(config: LogCompressorConfig) -> Self {
        Self { config }
    }

    pub fn compress(&self, content: &str, bias: f64) -> LogCompressionResult {
        let lines: Vec<&str> = content.split('\n').collect();
        let original_line_count = lines.len();

        if original_line_count < self.config.min_lines_to_compress {
            return LogCompressionResult {
                compressed: content.to_string(),
                original_line_count,
                compressed_line_count: original_line_count,
                format_detected: LogFormat::Generic,
                compression_ratio: 1.0,
            };
        }

        let format = detect_format(&lines);
        let log_lines = self.parse_lines(&lines);
        let selected = self.select_lines(&log_lines, bias);
        let compressed = self.format_output(&selected, &log_lines);
        let ratio = compressed.len() as f64 / content.len().max(1) as f64;

        LogCompressionResult {
            compressed,
            original_line_count,
            compressed_line_count: selected.len(),
            format_detected: format,
            compression_ratio: ratio,
        }
    }

    fn parse_lines(&self, lines: &[&str]) -> Vec<LogLine> {
        let mut out: Vec<LogLine> = Vec::with_capacity(lines.len());
        let mut active: Option<TraceFlavor> = None;
        let mut trace_lines = 0usize;

        for (i, line) in lines.iter().enumerate() {
            let mut entry = LogLine::new(i, *line);
            entry.level = classify_level(line);
            entry.is_summary = is_summary_line(line);

            if let Some(flavor) = active {
                if trace_lines >= self.config.stack_trace_max_lines || terminates(flavor, line) {
                    active = None;
                    trace_lines = 0;
                    if let Some(new_flavor) = flavor_for(line) {
                        active = Some(new_flavor);
                        trace_lines = 1;
                        entry.is_stack_trace = true;
                    }
                } else {
                    entry.is_stack_trace = true;
                    trace_lines += 1;
                }
            } else if let Some(flavor) = flavor_for(line) {
                active = Some(flavor);
                trace_lines = 1;
                entry.is_stack_trace = true;
            }

            entry.score = score_log_line(&entry);
            out.push(entry);
        }
        out
    }

    fn select_lines(&self, log_lines: &[LogLine], bias: f64) -> Vec<LogLine> {
        let all_strings: Vec<&str> = log_lines.iter().map(|l| l.content.as_str()).collect();
        let adaptive_max =
            compute_optimal_k(&all_strings, bias, 10, Some(self.config.max_total_lines));

        let mut errors: Vec<LogLine> = Vec::new();
        let mut fails: Vec<LogLine> = Vec::new();
        let mut warnings: Vec<LogLine> = Vec::new();
        let mut summaries: Vec<LogLine> = Vec::new();
        let mut stack_traces: Vec<Vec<LogLine>> = Vec::new();
        let mut current_stack: Vec<LogLine> = Vec::new();

        for line in log_lines {
            match line.level {
                LogLevel::Error => errors.push(line.clone()),
                LogLevel::Fail => fails.push(line.clone()),
                LogLevel::Warn => warnings.push(line.clone()),
                _ => {}
            }
            if line.is_stack_trace {
                current_stack.push(line.clone());
            } else if !current_stack.is_empty() {
                stack_traces.push(std::mem::take(&mut current_stack));
            }
            if line.is_summary {
                summaries.push(line.clone());
            }
        }
        if !current_stack.is_empty() {
            stack_traces.push(current_stack);
        }

        let mut selected: BTreeSet<LogLine> = BTreeSet::new();

        for line in self.select_with_first_last(&errors, self.config.max_errors) {
            selected.insert(line);
        }
        for line in self.select_with_first_last(&fails, self.config.max_errors) {
            selected.insert(line);
        }

        let warnings = if self.config.dedupe_warnings {
            dedupe_similar(warnings)
        } else {
            warnings
        };
        for line in warnings.into_iter().take(self.config.max_warnings) {
            selected.insert(line);
        }

        for stack in stack_traces.iter().take(self.config.max_stack_traces) {
            for line in stack.iter().take(self.config.stack_trace_max_lines) {
                selected.insert(line.clone());
            }
        }

        if self.config.keep_summary_lines {
            for line in summaries {
                selected.insert(line);
            }
        }

        let selected_indices: BTreeSet<usize> = selected.iter().map(|l| l.line_number).collect();
        let mut context_indices: BTreeSet<usize> = BTreeSet::new();
        for &idx in &selected_indices {
            let lo = idx.saturating_sub(self.config.error_context_lines);
            let hi = (idx + self.config.error_context_lines + 1).min(log_lines.len());
            for i in lo..hi {
                if i != idx {
                    context_indices.insert(i);
                }
            }
        }
        for idx in context_indices {
            if !selected_indices.contains(&idx) && idx < log_lines.len() {
                selected.insert(log_lines[idx].clone());
            }
        }

        let mut ordered: Vec<LogLine> = selected.into_iter().collect();
        if ordered.len() > adaptive_max {
            ordered.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.line_number.cmp(&b.line_number))
            });
            ordered.truncate(adaptive_max);
            ordered.sort_by_key(|l| l.line_number);
        }
        ordered
    }

    fn select_with_first_last(&self, lines: &[LogLine], max_count: usize) -> Vec<LogLine> {
        if lines.len() <= max_count {
            return lines.to_vec();
        }
        let mut out: Vec<LogLine> = Vec::with_capacity(max_count);
        let mut seen: BTreeSet<usize> = BTreeSet::new();
        if self.config.keep_first_error && seen.insert(lines[0].line_number) {
            out.push(lines[0].clone());
        }
        if self.config.keep_last_error {
            let last = lines.last().unwrap();
            if seen.insert(last.line_number) {
                out.push(last.clone());
            }
        }
        let remaining = max_count.saturating_sub(out.len());
        if remaining > 0 {
            let mut by_score = lines.to_vec();
            by_score.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.line_number.cmp(&b.line_number))
            });
            for line in by_score {
                if seen.insert(line.line_number) {
                    out.push(line);
                    if out.len() >= max_count {
                        break;
                    }
                }
            }
        }
        out
    }

    fn format_output(&self, selected: &[LogLine], all_lines: &[LogLine]) -> String {
        let mut output: Vec<String> = selected.iter().map(|l| l.content.clone()).collect();

        let omitted = all_lines.len().saturating_sub(selected.len());
        if omitted > 0 {
            let mut parts: Vec<String> = Vec::new();
            for (label, level) in [
                ("ERROR", LogLevel::Error),
                ("FAIL", LogLevel::Fail),
                ("WARN", LogLevel::Warn),
                ("INFO", LogLevel::Info),
            ] {
                let n = all_lines.iter().filter(|l| l.level == level).count();
                if n > 0 {
                    parts.push(format!("{} {}", n, label));
                }
            }
            if !parts.is_empty() {
                output.push(format!("[{} lines omitted: {}]", omitted, parts.join(", ")));
            }
        }
        output.join("\n")
    }
}

fn detect_format(lines: &[&str]) -> LogFormat {
    let table: &[(LogFormat, &[&str])] = &[
        (
            LogFormat::Pytest,
            &[
                "=== FAILURES",
                "=== ERRORS",
                "=== test session",
                "=== short test summary",
                "PASSED [",
                "FAILED [",
                "ERROR [",
                "SKIPPED [",
                "collected ",
            ],
        ),
        (
            LogFormat::Npm,
            &["npm ERR!", "npm WARN", "npm info", "npm http"],
        ),
        (
            LogFormat::Cargo,
            &[
                "Compiling ",
                "Finished ",
                "Running ",
                "warning: ",
                "error[E",
            ],
        ),
        (LogFormat::Jest, &["PASS ", "FAIL ", "Test Suites:"]),
        (
            LogFormat::Make,
            &["make[", "make:", "gcc ", "g++ ", "clang "],
        ),
    ];

    let sample: Vec<&str> = lines.iter().take(100).copied().collect();
    let mut best: Option<(LogFormat, usize)> = None;
    for (fmt, patterns) in table {
        let mut score = 0;
        for line in &sample {
            if patterns.iter().any(|p| line.contains(p)) {
                score += 1;
            }
        }
        if score > 0 && best.map(|(_, s)| score > s).unwrap_or(true) {
            best = Some((*fmt, score));
        }
    }
    best.map(|(f, _)| f).unwrap_or(LogFormat::Generic)
}

fn classify_level(line: &str) -> LogLevel {
    // Order matters: error/fail before warn before info. Word-boundary aware.
    const ENTRIES: &[(LogLevel, &[&str])] = &[
        (LogLevel::Error, &["ERROR", "FATAL", "CRITICAL"]),
        (LogLevel::Fail, &["FAILED", "FAIL"]),
        (LogLevel::Warn, &["WARNING", "WARN"]),
        (LogLevel::Info, &["INFO"]),
        (LogLevel::Debug, &["DEBUG"]),
        (LogLevel::Trace, &["TRACE"]),
    ];
    let bytes = line.as_bytes();
    let mut best: Option<(usize, LogLevel)> = None;
    for (level, words) in ENTRIES {
        for w in *words {
            if let Some(pos) = find_word_ci(line, w) {
                let end = pos + w.len();
                if is_word_boundary(bytes, pos, end) {
                    match best {
                        Some((bp, _)) if bp <= pos => {}
                        _ => best = Some((pos, *level)),
                    }
                }
            }
        }
    }
    best.map(|(_, l)| l).unwrap_or(LogLevel::Unknown)
}

/// Case-insensitive search for the leftmost occurrence of `needle` in `hay`.
fn find_word_ci(hay: &str, needle: &str) -> Option<usize> {
    let hb = hay.as_bytes();
    let nb = needle.as_bytes();
    if nb.is_empty() || nb.len() > hb.len() {
        return None;
    }
    (0..=hb.len() - nb.len()).find(|&i| {
        hb[i..i + nb.len()]
            .iter()
            .zip(nb)
            .all(|(a, b)| a.eq_ignore_ascii_case(b))
    })
}

fn is_word_boundary(bytes: &[u8], start: usize, end: usize) -> bool {
    let left_ok = start == 0 || !is_word_byte(bytes[start - 1]);
    let right_ok = end == bytes.len() || !is_word_byte(bytes[end]);
    left_ok && right_ok
}

#[inline]
fn is_word_byte(b: u8) -> bool {
    matches!(b, b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_')
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TraceFlavor {
    PythonTraceback,
    Js,
    Java,
    RustError,
    Go,
}

fn flavor_for(line: &str) -> Option<TraceFlavor> {
    let trimmed = line.trim_start();
    if trimmed.starts_with("Traceback (most recent call last)") || is_python_file_frame(trimmed) {
        Some(TraceFlavor::PythonTraceback)
    } else if is_js_at_frame(trimmed) {
        Some(TraceFlavor::Js)
    } else if is_java_at_frame(trimmed) {
        Some(TraceFlavor::Java)
    } else if trimmed.starts_with("--> ") && has_line_col_suffix(trimmed) {
        Some(TraceFlavor::RustError)
    } else if is_go_frame(line) {
        Some(TraceFlavor::Go)
    } else {
        None
    }
}

fn is_python_file_frame(s: &str) -> bool {
    s.starts_with("File \"")
        && s.contains("\", line ")
        && s.bytes().next_back().is_some_and(|b| b.is_ascii_digit())
}

fn is_js_at_frame(s: &str) -> bool {
    s.starts_with("at ") && s.contains('(') && s.contains(')') && has_line_col_suffix(s)
}

fn is_java_at_frame(s: &str) -> bool {
    let Some(after_at) = s.strip_prefix("at ") else {
        return false;
    };
    if !after_at.contains('(') {
        return false;
    }
    let body = after_at.split('(').next().unwrap_or("");
    !body.is_empty()
        && body
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '$'))
}

fn has_line_col_suffix(s: &str) -> bool {
    let bytes = s.as_bytes();
    for i in 0..bytes.len().saturating_sub(2) {
        if bytes[i] == b':' && bytes[i + 1].is_ascii_digit() {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            if j < bytes.len()
                && bytes[j] == b':'
                && bytes.get(j + 1).is_some_and(|b| b.is_ascii_digit())
            {
                return true;
            }
        }
    }
    false
}

fn is_go_frame(s: &str) -> bool {
    let trimmed = s.trim_start();
    let mut chars = trimmed.chars().peekable();
    let mut saw_digit = false;
    while let Some(&c) = chars.peek() {
        if c.is_ascii_digit() {
            saw_digit = true;
            chars.next();
        } else {
            break;
        }
    }
    if !saw_digit || chars.next() != Some(':') {
        return false;
    }
    while chars.peek() == Some(&' ') {
        chars.next();
    }
    let rest: String = chars.collect();
    rest.strip_prefix("0x")
        .is_some_and(|hex| hex.chars().take_while(|c| c.is_ascii_hexdigit()).count() > 0)
}

fn terminates(flavor: TraceFlavor, line: &str) -> bool {
    let trimmed = line.trim_start();
    match flavor {
        TraceFlavor::PythonTraceback => {
            let is_indented_or_blank = line.starts_with([' ', '\t']) || line.is_empty();
            let is_continuation = trimmed.starts_with("Traceback")
                || trimmed.starts_with("File ")
                || trimmed.starts_with("During handling")
                || trimmed.starts_with("The above exception");
            if is_indented_or_blank || is_continuation {
                false
            } else {
                !trimmed.starts_with(char::is_uppercase)
            }
        }
        TraceFlavor::Js | TraceFlavor::Java => !trimmed.starts_with("at ") && !line.is_empty(),
        TraceFlavor::RustError => !trimmed.starts_with("--> ") && !line.is_empty(),
        TraceFlavor::Go => {
            !trimmed.chars().next().is_some_and(|c| c.is_ascii_digit()) && !line.is_empty()
        }
    }
}

fn is_summary_line(line: &str) -> bool {
    if line.starts_with("===") || line.starts_with("---") {
        return true;
    }
    let bytes = line.as_bytes();
    let leading_digits = bytes.iter().take_while(|b| b.is_ascii_digit()).count();
    if leading_digits > 0 {
        let (_, after_digits) = line.split_at(leading_digits);
        if let Some(rest) = after_digits.strip_prefix(' ') {
            for kw in &["passed", "failed", "skipped", "error", "warning"] {
                if rest.starts_with(kw) {
                    return true;
                }
            }
        }
    }
    for prefix in &[
        "Test ", "Tests ", "Tests:", "Test:", "Suite ", "Suites ", "Suites:", "Suite:",
    ] {
        if let Some(rest) = line.strip_prefix(prefix) {
            return rest
                .chars()
                .find(|c| !c.is_whitespace())
                .is_some_and(|c| c.is_ascii_digit());
        }
    }
    for prefix in &["TOTAL", "Total", "Summary"] {
        if line.starts_with(prefix) {
            return true;
        }
    }
    for prefix in &["Build", "Compile", "Test"] {
        if line.starts_with(prefix) {
            for outcome in &["succeeded", "failed", "complete"] {
                if line.contains(outcome) {
                    return true;
                }
            }
        }
    }
    false
}

fn score_log_line(line: &LogLine) -> f32 {
    let level_score: f32 = match line.level {
        LogLevel::Error | LogLevel::Fail => 1.0,
        LogLevel::Warn => 0.5,
        LogLevel::Info | LogLevel::Unknown => 0.1,
        LogLevel::Debug => 0.05,
        LogLevel::Trace => 0.02,
    };
    let stack_boost = if line.is_stack_trace { 0.3 } else { 0.0 };
    let summary_boost = if line.is_summary { 0.4 } else { 0.0 };
    (level_score + stack_boost + summary_boost).min(1.0)
}

fn dedupe_similar(lines: Vec<LogLine>) -> Vec<LogLine> {
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut out: Vec<LogLine> = Vec::with_capacity(lines.len());
    for line in lines {
        let key = normalize_for_dedupe(&line.content);
        if seen.insert(key) {
            out.push(line);
        }
    }
    out
}

/// Conservative normalizer for warning dedup: preserves the message prefix
/// (everything before the first `:` or `=`) and only normalizes the trailing
/// variable region (digits, hex addresses, paths).
fn normalize_for_dedupe(content: &str) -> String {
    let split_pos = content.find([':', '=']).unwrap_or(content.len());
    let (prefix, suffix) = content.split_at(split_pos);

    let stage1 = hex_regex().replace_all(suffix, "ADDR");
    let stage2 = digit_regex().replace_all(&stage1, "N");
    let stage3 = path_regex().replace_all(&stage2, "/PATH/");
    format!("{}{}", prefix, stage3)
}

fn digit_regex() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"\d+").unwrap())
}
fn hex_regex() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"0x[0-9a-fA-F]+").unwrap())
}
fn path_regex() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"/[\w/]+/").unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cmp() -> LogCompressor {
        LogCompressor::new(LogCompressorConfig::default())
    }

    #[test]
    fn short_logs_pass_through() {
        let content = "line1\nline2\nERROR boom";
        let r = cmp().compress(content, 1.0);
        assert_eq!(r.compressed, content);
        assert_eq!(r.compression_ratio, 1.0);
    }

    #[test]
    fn compresses_noisy_build_output_and_keeps_errors() {
        let mut lines: Vec<String> = Vec::new();
        for i in 0..400 {
            lines.push(format!("INFO compiling module_{i} ... ok"));
        }
        lines.push("ERROR: undefined reference to `frobnicate`".to_string());
        for i in 0..200 {
            lines.push(format!("INFO linking object_{i} ... ok"));
        }
        let content = lines.join("\n");

        let r = cmp().compress(&content, 1.0);
        assert!(
            r.compressed_line_count < r.original_line_count,
            "should drop lines"
        );
        assert!(
            r.compressed.contains("undefined reference to `frobnicate`"),
            "the single error must survive: {}",
            r.compressed
        );
        assert!(r.compression_ratio < 0.5, "ratio={}", r.compression_ratio);
        assert!(r.compressed.contains("lines omitted"));
    }

    #[test]
    fn keeps_python_traceback_together() {
        let mut lines: Vec<String> = (0..60).map(|i| format!("DEBUG step {i}")).collect();
        lines.push("Traceback (most recent call last):".to_string());
        lines.push("  File \"app.py\", line 10".to_string());
        lines.push("    do_thing()".to_string());
        lines.push("ValueError: bad value".to_string());
        let content = lines.join("\n");

        let r = cmp().compress(&content, 1.0);
        assert!(r.compressed.contains("Traceback (most recent call last)"));
        assert!(r.compressed.contains("ValueError: bad value"));
    }

    #[test]
    fn detects_pytest_format() {
        let lines = [
            "============================= test session starts =============================",
            "collected 15 items",
            "tests/test_foo.py::test_basic PASSED [  6%]",
            "FAILED tests/test_foo.py::test_edge",
        ];
        assert_eq!(detect_format(&lines), LogFormat::Pytest);
    }

    #[test]
    fn dedupes_similar_warnings_preserving_distinct_messages() {
        let warnings = vec![
            LogLine::new(1, "WARNING: deprecated call at 0xdead"),
            LogLine::new(2, "WARNING: deprecated call at 0xbeef"),
            LogLine::new(3, "WARNING: unused variable x"),
        ];
        let deduped = dedupe_similar(warnings);
        assert_eq!(deduped.len(), 2, "addr-only diffs collapse, message stays");
    }
}
