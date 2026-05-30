//! Pre-flight router that scores a conversation's cognitive complexity in [0, 1]
//! so the agent can decide whether to send a turn to the fast model.
//!
//! `ComplexityModel` loads a self-contained bundle from `~/.goose/complexity_model/`
//! containing a fastembed-style ONNX embedder, an HF tokenizer, an MLP head
//! exported as safetensors, and a `config.json` describing the architecture.
//!
//! On top of the model this module exposes the small bits of policy the agent
//! needs: rendering a `Conversation` into the same text format the model was
//! trained on, and the threshold logic for "is this a fast-model turn?"
//!
//! See `GOOSE_INTEGRATION_PLAN.md` in the llm-router repo for the design;
//! the bundle is produced by `train_complexity.py`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use ndarray::{Array1, Array2};
use ort::session::Session;
use ort::value::TensorRef;
use safetensors::SafeTensors;
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::conversation::message::{Message, Role};
use crate::conversation::Conversation;

const DEFAULT_BUNDLE_SUBDIR: &str = "complexity_model";
const MAX_SEQ_LEN: usize = 512;
const ANCHOR_MARKER: &str = ">>>";

/// Complexity threshold under which the agent routes the turn to the fast model.
/// Same dial we used at training time; tuned empirically by eyeballing the
/// WildChat distribution. Demo-friendly default.
pub const FAST_MODEL_THRESHOLD: f32 = 0.5;

#[derive(Debug, Deserialize)]
struct BundleConfig {
    format_version: u32,
    embedder: EmbedderConfig,
    head: HeadConfig,
}

#[derive(Debug, Deserialize)]
struct EmbedderConfig {
    repo_id: String,
    output_dim: usize,
    onnx_file: String,
    tokenizer_file: String,
}

#[derive(Debug, Deserialize)]
struct HeadConfig {
    input_dim: usize,
    hidden_dims: Vec<usize>,
    output_dim: usize,
    #[serde(default)]
    activation: String,
    #[serde(default)]
    output_activation: String,
}

/// One linear layer worth of weights, as owned `ndarray` matrices.
struct LinearLayer {
    weight: Array2<f32>, // (out_dim, in_dim) — matches torch.nn.Linear convention
    bias: Array1<f32>,   // (out_dim,)
}

impl LinearLayer {
    fn apply(&self, x: &Array1<f32>) -> Array1<f32> {
        self.weight.dot(x) + &self.bias
    }
}

/// Output of a single complexity scoring call.
#[derive(Debug, Clone, Copy)]
pub struct ComplexityScore {
    pub complexity: f32,
    pub tool_calls_norm: f32,
    pub elapsed_ms: u64,
}

/// Loaded complexity model. Cheap to clone (everything inside an `Arc`).
pub struct ComplexityModel {
    inner: Arc<Inner>,
}

struct Inner {
    embedder_dim: usize,
    tokenizer: Tokenizer,
    session: Session,
    head: Vec<LinearLayer>,
    head_out_dim: usize,
    repo_id: String,
}

impl ComplexityModel {
    /// Default location: `~/.goose/complexity_model/`. Returns `None` (not an
    /// error) if the bundle is missing — callers should treat that as
    /// "complexity routing disabled."
    pub fn try_load_default() -> Option<Self> {
        let dir = default_bundle_dir()?;
        if !dir.join("config.json").exists() {
            return None;
        }
        match Self::load_from_dir(&dir) {
            Ok(m) => Some(m),
            Err(e) => {
                tracing::warn!("failed to load complexity model from {:?}: {:#}", dir, e);
                None
            }
        }
    }

    pub fn load_from_dir(dir: &Path) -> Result<Self> {
        let cfg_path = dir.join("config.json");
        let cfg_text = std::fs::read_to_string(&cfg_path)
            .with_context(|| format!("reading {}", cfg_path.display()))?;
        let cfg: BundleConfig = serde_json::from_str(&cfg_text)
            .with_context(|| format!("parsing {}", cfg_path.display()))?;

        if cfg.format_version != 1 {
            bail!("unsupported bundle format_version {}", cfg.format_version);
        }
        if cfg.head.input_dim != cfg.embedder.output_dim {
            bail!(
                "head input_dim {} != embedder output_dim {}",
                cfg.head.input_dim,
                cfg.embedder.output_dim
            );
        }

        let tokenizer_path = dir.join(&cfg.embedder.tokenizer_file);
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("loading tokenizer at {}: {}", tokenizer_path.display(), e))?;

        let onnx_path = dir.join(&cfg.embedder.onnx_file);
        let session = Session::builder()?
            .commit_from_file(&onnx_path)
            .with_context(|| format!("loading ONNX from {}", onnx_path.display()))?;

        let weights_path = dir.join("weights.safetensors");
        let weights_bytes = std::fs::read(&weights_path)
            .with_context(|| format!("reading {}", weights_path.display()))?;
        let head = load_head_weights(&weights_bytes, &cfg.head)?;

        Ok(Self {
            inner: Arc::new(Inner {
                embedder_dim: cfg.embedder.output_dim,
                tokenizer,
                session,
                head,
                head_out_dim: cfg.head.output_dim,
                repo_id: cfg.embedder.repo_id,
            }),
        })
    }

    /// Score one rendered conversation. ~25-50ms on CPU for typical inputs.
    pub fn score(&self, text: &str) -> Result<ComplexityScore> {
        let started = Instant::now();
        let embedding = self.embed(text)?;
        let mut activations = embedding;
        let last = self.inner.head.len() - 1;
        for (i, layer) in self.inner.head.iter().enumerate() {
            activations = layer.apply(&activations);
            if i != last {
                activations.mapv_inplace(|x| x.max(0.0)); // ReLU
            }
        }
        activations.mapv_inplace(sigmoid);

        if activations.len() < self.inner.head_out_dim {
            bail!(
                "head output has {} values, expected {}",
                activations.len(),
                self.inner.head_out_dim
            );
        }

        Ok(ComplexityScore {
            complexity: activations[0],
            tool_calls_norm: if self.inner.head_out_dim >= 2 {
                activations[1]
            } else {
                0.0
            },
            elapsed_ms: started.elapsed().as_millis() as u64,
        })
    }

    pub fn embedder_repo(&self) -> &str {
        &self.inner.repo_id
    }

    fn embed(&self, text: &str) -> Result<Array1<f32>> {
        let encoding = self
            .inner
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("tokenize: {}", e))?;
        let mut ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let mut mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&x| x as i64)
            .collect();
        if ids.len() > MAX_SEQ_LEN {
            ids.truncate(MAX_SEQ_LEN);
            mask.truncate(MAX_SEQ_LEN);
        }
        let seq_len = ids.len();
        let ids_arr = Array2::from_shape_vec((1, seq_len), ids)?;
        let mask_arr = Array2::from_shape_vec((1, seq_len), mask)?;
        // bge-m3 is XLM-RoBERTa under the hood — single segment, so token_type_ids
        // is all zeros. The ONNX input is required nonetheless.
        let type_arr = Array2::<i64>::zeros((1, seq_len));

        let outputs = self.inner.session.run(ort::inputs![
            "input_ids" => TensorRef::from_array_view(ids_arr.view())?,
            "attention_mask" => TensorRef::from_array_view(mask_arr.view())?,
            "token_type_ids" => TensorRef::from_array_view(type_arr.view())?,
        ])?;

        // bge-m3 ONNX outputs `last_hidden_state` of shape (1, seq_len, hidden).
        // We CLS-pool: take position 0.
        let last_hidden = outputs
            .get("last_hidden_state")
            .ok_or_else(|| anyhow!("ONNX output is missing 'last_hidden_state'"))?;
        let tensor_view = last_hidden.try_extract_array::<f32>()?;
        let shape = tensor_view.shape();
        if shape.len() != 3 {
            bail!("expected 3D embedder output, got shape {:?}", shape);
        }
        let hidden = shape[2];
        if hidden != self.inner.embedder_dim {
            bail!(
                "embedder hidden dim {} != config output_dim {}",
                hidden,
                self.inner.embedder_dim
            );
        }
        // CLS token = position 0 along the sequence axis.
        let cls_view = tensor_view.slice(ndarray::s![0, 0, ..]);
        let mut emb = Array1::from_iter(cls_view.iter().copied());

        // bge-m3 outputs are unit-normalized in fastembed; do the same here.
        let norm = emb.dot(&emb).sqrt();
        if norm > 1e-12 {
            emb.mapv_inplace(|v| v / norm);
        }
        Ok(emb)
    }
}

fn default_bundle_dir() -> Option<PathBuf> {
    let home = dirs::home_dir()?;
    Some(home.join(".goose").join(DEFAULT_BUNDLE_SUBDIR))
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn load_head_weights(bytes: &[u8], cfg: &HeadConfig) -> Result<Vec<LinearLayer>> {
    let st = SafeTensors::deserialize(bytes).context("parse safetensors")?;

    // The Python head is `Sequential(Linear, ReLU, Dropout, Linear, ReLU, Dropout, ...)`
    // followed by `Linear` named `out`. Inside `Sequential` the linears are at
    // indices 0, 3, 6, … which corresponds to `trunk.{0,3,6,…}` in the
    // state_dict. We materialize them in order plus the final `out` linear.
    let mut layers = Vec::with_capacity(cfg.hidden_dims.len() + 1);
    let mut sequential_idx = 0usize;
    let mut prev_dim = cfg.input_dim;
    for &h in &cfg.hidden_dims {
        let w_name = format!("trunk.{}.weight", sequential_idx);
        let b_name = format!("trunk.{}.bias", sequential_idx);
        layers.push(read_linear(&st, &w_name, &b_name, h, prev_dim)?);
        prev_dim = h;
        sequential_idx += 3; // skip ReLU + Dropout
    }
    layers.push(read_linear(
        &st,
        "out.weight",
        "out.bias",
        cfg.output_dim,
        prev_dim,
    )?);
    Ok(layers)
}

fn read_linear(
    st: &SafeTensors,
    weight_name: &str,
    bias_name: &str,
    expected_out: usize,
    expected_in: usize,
) -> Result<LinearLayer> {
    let w_tensor = st
        .tensor(weight_name)
        .with_context(|| format!("missing tensor {}", weight_name))?;
    let b_tensor = st
        .tensor(bias_name)
        .with_context(|| format!("missing tensor {}", bias_name))?;

    let w_shape = w_tensor.shape();
    if w_shape != [expected_out, expected_in] {
        bail!(
            "{}: shape {:?} != expected [{}, {}]",
            weight_name,
            w_shape,
            expected_out,
            expected_in
        );
    }
    let b_shape = b_tensor.shape();
    if b_shape != [expected_out] {
        bail!(
            "{}: shape {:?} != expected [{}]",
            bias_name,
            b_shape,
            expected_out
        );
    }

    let w_data = bytes_to_f32(w_tensor.data())?;
    let b_data = bytes_to_f32(b_tensor.data())?;
    let weight = Array2::from_shape_vec((expected_out, expected_in), w_data)?;
    let bias = Array1::from_vec(b_data);
    Ok(LinearLayer { weight, bias })
}

fn bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        bail!("tensor byte length {} is not a multiple of 4", bytes.len());
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

/// Render `conversation` into the `user:/assistant:` form used at training
/// time. The most recent user message is the anchor, marked with `>>>`;
/// everything before it is the context.
///
/// Returns `None` if no user message exists. We don't enforce a token budget
/// here — the embedder truncates to its max sequence length, and the anchor
/// is at the end so truncation drops the oldest context first.
pub fn render_for_routing(conversation: &Conversation) -> Option<String> {
    let messages = conversation.messages();
    let anchor_idx = messages
        .iter()
        .enumerate()
        .rev()
        .find(|(_, m)| matches!(m.role, Role::User))
        .map(|(i, _)| i)?;

    let mut lines = Vec::with_capacity(anchor_idx + 1);
    for (i, msg) in messages.iter().take(anchor_idx + 1).enumerate() {
        let role = match msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        };
        let text = msg.as_concat_text();
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        if i == anchor_idx {
            lines.push(format!("{} {}: {}", ANCHOR_MARKER, role, text));
        } else {
            lines.push(format!("{}: {}", role, text));
        }
    }
    if lines.is_empty() {
        None
    } else {
        Some(lines.join("\n"))
    }
}

/// All-in-one: render a conversation, score it, decide whether to use the
/// fast model. Returns `None` if the conversation has no user turn (which
/// would be weird).
pub fn route(model: &ComplexityModel, conversation: &Conversation) -> Option<RouteDecision> {
    let rendered = render_for_routing(conversation)?;
    match model.score(&rendered) {
        Ok(score) => Some(RouteDecision {
            complexity: score.complexity,
            use_fast: score.complexity < FAST_MODEL_THRESHOLD,
            elapsed_ms: score.elapsed_ms,
        }),
        Err(e) => {
            tracing::warn!("complexity scoring failed, defaulting to smart model: {:#}", e);
            None
        }
    }
}

/// What the agent loop needs to know.
#[derive(Debug, Clone, Copy)]
pub struct RouteDecision {
    pub complexity: f32,
    pub use_fast: bool,
    pub elapsed_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::message::Message;

    fn convo(msgs: Vec<Message>) -> Conversation {
        Conversation::new_unvalidated(msgs)
    }

    #[test]
    fn render_empty_returns_none() {
        let c = convo(vec![]);
        assert!(render_for_routing(&c).is_none());
    }

    #[test]
    fn render_single_user_message() {
        let c = convo(vec![Message::user().with_text("hi there")]);
        let r = render_for_routing(&c).expect("some");
        assert_eq!(r, ">>> user: hi there");
    }

    #[test]
    fn render_multi_turn_marks_last_user() {
        let c = convo(vec![
            Message::user().with_text("what is 2+2"),
            Message::assistant().with_text("4"),
            Message::user().with_text("now squared"),
        ]);
        let r = render_for_routing(&c).expect("some");
        assert_eq!(
            r,
            "user: what is 2+2\nassistant: 4\n>>> user: now squared"
        );
    }

    #[test]
    fn render_skips_empty_messages() {
        let c = convo(vec![
            Message::user().with_text("real question"),
            Message::assistant().with_text(""),
            Message::user().with_text("follow up"),
        ]);
        let r = render_for_routing(&c).expect("some");
        assert_eq!(r, "user: real question\n>>> user: follow up");
    }
}
