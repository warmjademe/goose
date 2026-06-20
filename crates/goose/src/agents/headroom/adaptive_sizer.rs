//! Adaptive compression sizing via information-saturation detection.
//!
//! Port of headroom's `adaptive_sizer` (Kneedle on a cumulative
//! unique-bigram coverage curve, with a SimHash redundancy fast-path and a
//! zlib-ratio validation pass). Decides *how many* items to keep from an
//! importance-ordered list.

use flate2::write::ZlibEncoder;
use flate2::Compression;
use md5::{Digest, Md5};
use std::collections::HashSet;
use std::io::Write;

/// Compute the optimal number of items to keep via information saturation.
///
/// - `items`: string representations of items in importance order.
/// - `bias`: multiplier on the knee point (>1 keeps more, <1 compresses harder).
/// - `min_k`: lower bound on the return value.
/// - `max_k`: upper bound; `None` means "up to `items.len()`".
pub fn compute_optimal_k(items: &[&str], bias: f64, min_k: usize, max_k: Option<usize>) -> usize {
    let n = items.len();
    let effective_max = max_k.unwrap_or(n);

    if n <= 8 {
        return n;
    }

    let unique_count = count_unique_simhash(items, 3);
    if unique_count <= 3 {
        let k = min_k.max(unique_count);
        return k.min(effective_max);
    }

    let curve = compute_unique_bigram_curve(items);
    let mut knee = find_knee(&curve);

    let diversity_ratio = unique_count as f64 / n as f64;

    knee = match knee {
        None => {
            let keep_fraction = 0.3 + 0.7 * diversity_ratio;
            Some(min_k.max((n as f64 * keep_fraction) as usize))
        }
        Some(k) if diversity_ratio > 0.7 => {
            let floor = min_k.max((n as f64 * (0.3 + 0.7 * diversity_ratio)) as usize);
            Some(k.max(floor))
        }
        some => some,
    };

    let knee = knee.unwrap_or(min_k);

    let mut k = min_k.max((knee as f64 * bias) as usize);
    k = k.min(effective_max);

    k = validate_with_zlib(items, k, effective_max, 0.15);

    min_k.max(k.min(effective_max))
}

/// Find the knee in a monotonically-increasing curve (Kneedle). Returns a
/// 1-indexed "keep this many" count.
pub fn find_knee(curve: &[usize]) -> Option<usize> {
    let n = curve.len();
    if n < 3 {
        return None;
    }

    let x_max = n - 1;
    let y_min = curve[0] as f64;
    let y_max = curve[n - 1] as f64;

    if (y_max - y_min).abs() < f64::EPSILON {
        return Some(1);
    }

    let x_range = x_max as f64;
    let y_range = y_max - y_min;

    let mut max_diff: f64 = -1.0;
    let mut knee_idx: Option<usize> = None;

    for (i, &y) in curve.iter().enumerate() {
        let x_norm = i as f64 / x_range;
        let y_norm = (y as f64 - y_min) / y_range;
        let diff = y_norm - x_norm;
        if diff > max_diff {
            max_diff = diff;
            knee_idx = Some(i);
        }
    }

    if max_diff < 0.05 {
        return None;
    }

    knee_idx.map(|i| i + 1)
}

/// Cumulative unique-word-bigram coverage curve.
pub fn compute_unique_bigram_curve(items: &[&str]) -> Vec<usize> {
    let mut seen: HashSet<(String, String)> = HashSet::new();
    let mut curve: Vec<usize> = Vec::with_capacity(items.len());

    for item in items {
        let lower = item.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();
        if words.len() < 2 {
            let first = words.first().copied().unwrap_or("");
            seen.insert((first.to_string(), String::new()));
        } else {
            for j in 0..words.len() - 1 {
                seen.insert((words[j].to_string(), words[j + 1].to_string()));
            }
        }
        curve.push(seen.len());
    }

    curve
}

/// 64-bit SimHash fingerprint over character 4-grams.
pub fn simhash(text: &str) -> u64 {
    let lower = text.to_lowercase();
    let chars: Vec<char> = lower.chars().collect();
    let n = chars.len();

    let iter_count = if n <= 3 { 1 } else { n - 3 };
    let mut votes: [i32; 64] = [0; 64];

    for i in 0..iter_count {
        let gram: String = chars.iter().skip(i).take(4).collect();
        let digest = Md5::digest(gram.as_bytes());
        let h = u64::from_be_bytes([
            digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6], digest[7],
        ]);
        for (j, vote) in votes.iter_mut().enumerate() {
            if (h >> j) & 1 == 1 {
                *vote += 1;
            } else {
                *vote -= 1;
            }
        }
    }

    let mut fingerprint: u64 = 0;
    for (j, &v) in votes.iter().enumerate() {
        if v > 0 {
            fingerprint |= 1 << j;
        }
    }
    fingerprint
}

#[inline]
pub fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Count items with distinct content via SimHash + greedy clustering.
pub fn count_unique_simhash(items: &[&str], threshold: u32) -> usize {
    if items.is_empty() {
        return 0;
    }

    let fingerprints: Vec<u64> = items.iter().map(|s| simhash(s)).collect();
    let mut clusters: Vec<u64> = Vec::new();

    for &fp in &fingerprints {
        let matched = clusters
            .iter()
            .any(|&rep| hamming_distance(fp, rep) <= threshold);
        if !matched {
            clusters.push(fp);
        }
    }

    clusters.len()
}

/// zlib-based compression-ratio validation. If the chosen subset compresses
/// much better than the full set, it is missing diversity → bump `k` 20%.
pub fn validate_with_zlib(items: &[&str], k: usize, max_k: usize, tolerance: f64) -> usize {
    if k >= items.len() || k >= max_k {
        return k;
    }

    let full_text = items.join("\n");
    let subset_text = items[..k].join("\n");

    if full_text.len() < 200 {
        return k;
    }

    let full_compressed = zlib_compressed_len(full_text.as_bytes());
    let subset_compressed = zlib_compressed_len(subset_text.as_bytes());

    let full_ratio = if !full_text.is_empty() {
        full_compressed as f64 / full_text.len() as f64
    } else {
        1.0
    };
    let subset_ratio = if !subset_text.is_empty() {
        subset_compressed as f64 / subset_text.len() as f64
    } else {
        1.0
    };

    if (full_ratio - subset_ratio).abs() > tolerance {
        let adjusted = ((k as f64) * 1.2) as usize;
        return adjusted.min(max_k);
    }

    k
}

fn zlib_compressed_len(bytes: &[u8]) -> usize {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::fast());
    encoder.write_all(bytes).expect("in-memory write");
    encoder.finish().expect("flush").len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simhash_empty_string() {
        assert_eq!(simhash(""), 0xd41d8cd98f00b204);
    }

    #[test]
    fn small_input_keeps_all() {
        let items = ["a", "b", "c"];
        assert_eq!(compute_optimal_k(&items, 1.0, 1, None), 3);
    }

    #[test]
    fn near_total_redundancy_collapses() {
        let items: Vec<String> = (0..50).map(|_| "identical line".to_string()).collect();
        let refs: Vec<&str> = items.iter().map(|s| s.as_str()).collect();
        let k = compute_optimal_k(&refs, 1.0, 1, Some(100));
        assert!(k <= 3, "near-identical items should collapse, got {k}");
    }

    #[test]
    fn diverse_input_keeps_more_than_redundant() {
        let diverse: Vec<String> = (0..50)
            .map(|i| format!("unique line number {i} xyz"))
            .collect();
        let diverse_refs: Vec<&str> = diverse.iter().map(|s| s.as_str()).collect();
        let redundant: Vec<String> = (0..50).map(|_| "same same same".to_string()).collect();
        let redundant_refs: Vec<&str> = redundant.iter().map(|s| s.as_str()).collect();

        let kd = compute_optimal_k(&diverse_refs, 1.0, 5, Some(100));
        let kr = compute_optimal_k(&redundant_refs, 1.0, 5, Some(100));
        assert!(kd > kr, "diverse {kd} should keep more than redundant {kr}");
    }
}
