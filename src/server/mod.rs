//! HTTP API server for BitNet-Oxidized inference.

use crate::errors::BitNetError;
use crate::model::{from_pretrained, gguf, ModelRegistry};
use crate::{BitNetTokenizer, Telemetry, TextGenerator};
use axum::{
    extract::State,
    http::{header, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

#[derive(Clone)]
pub struct ServerState {
    pub generator: Arc<RwLock<TextGenerator>>,
    pub tokenizer: Option<Arc<BitNetTokenizer>>,
    pub config: ServerConfig,
    pub telemetry: Option<Arc<Telemetry>>,
}

#[derive(Clone)]
pub struct ServerConfig {
    pub max_batch_size: usize,
    pub max_queue_size: usize,
    pub timeout_seconds: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 4,
            max_queue_size: 64,
            timeout_seconds: 30,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct GenerationRequest {
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub stream: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct GenerationResponse {
    pub text: String,
    pub tokens: Vec<usize>,
    pub finish_reason: String,
    pub usage: UsageStats,
}

#[derive(Debug, Serialize)]
pub struct UsageStats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub time_ms: f64,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub stream: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub choices: Vec<ChatChoice>,
    pub usage: UsageStats,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub owned_by: String,
}

#[derive(Debug, Serialize)]
pub struct ModelsListResponse {
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
pub struct MetricsResponse {
    pub requests_total: u64,
    pub tokens_generated_total: u64,
}

/// Run the HTTP server by model name or path (registry). If tokenizer_path is None or invalid,
/// uses simple word-based tokenization. If telemetry is Some, /metrics returns Prometheus text.
pub async fn run_server_with_registry(
    model_name_or_path: &str,
    registry: &ModelRegistry,
    tokenizer_path: Option<&Path>,
    port: u16,
    config: ServerConfig,
    telemetry: Option<Arc<Telemetry>>,
) -> Result<(), BitNetError> {
    let model = from_pretrained(model_name_or_path, registry)
        .map_err(|e| BitNetError::InvalidInput(e.to_string()))?;
    let generator = TextGenerator::new(model);
    let tokenizer: Option<Arc<BitNetTokenizer>> = match tokenizer_path {
        Some(p) => BitNetTokenizer::from_file(p).ok().map(Arc::new),
        None => None,
    };
    let state = Arc::new(ServerState {
        generator: Arc::new(RwLock::new(generator)),
        tokenizer,
        config,
        telemetry,
    });
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);
    let app = Router::new()
        .route("/", get(health_check))
        .route("/health", get(health_check))
        .route("/ui", get(serve_ui))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/metrics", get(metrics))
        .with_state(state)
        .layer(TraceLayer::new_for_http())
        .layer(cors);
    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(BitNetError::Io)?;
    tracing::info!("Server running on http://{}", addr);
    tracing::info!("Web UI: http://{}:{}/ui", addr.ip(), addr.port());
    axum::serve(listener, app).await.map_err(BitNetError::Io)?;
    Ok(())
}

/// Run the HTTP server by path. If path does not exist, uses demo model.
/// If tokenizer_path is None or invalid, uses simple word-based tokenization.
/// If telemetry is Some, /metrics returns Prometheus text format and requests are recorded.
pub async fn run_server(
    model_path: impl AsRef<Path>,
    tokenizer_path: Option<&Path>,
    port: u16,
    config: ServerConfig,
    telemetry: Option<Arc<Telemetry>>,
) -> Result<(), BitNetError> {
    let model = if model_path.as_ref().exists() {
        gguf::load_gguf(model_path.as_ref())
            .map_err(|e| BitNetError::InvalidFormat(e.to_string()))?
    } else {
        crate::model::create_demo_model()
    };

    let generator = TextGenerator::new(model);
    let tokenizer: Option<Arc<BitNetTokenizer>> = match tokenizer_path {
        Some(p) => BitNetTokenizer::from_file(p).ok().map(Arc::new),
        None => None,
    };

    let state = Arc::new(ServerState {
        generator: Arc::new(RwLock::new(generator)),
        tokenizer,
        config,
        telemetry,
    });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/", get(health_check))
        .route("/health", get(health_check))
        .route("/ui", get(serve_ui))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/metrics", get(metrics))
        .with_state(state)
        .layer(TraceLayer::new_for_http())
        .layer(cors);

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(BitNetError::Io)?;
    tracing::info!("Server running on http://{}", addr);
    tracing::info!("Web UI: http://{}:{}/ui", addr.ip(), addr.port());
    axum::serve(listener, app).await.map_err(BitNetError::Io)?;
    Ok(())
}

/// Serves the embedded web UI (prompt + completion).
async fn serve_ui() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        include_str!("../../static/index.html"),
    )
}

async fn health_check() -> &'static str {
    "OK"
}

async fn completions(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<GenerationRequest>,
) -> Result<Json<GenerationResponse>, ApiError> {
    let max_tokens = req.max_tokens.unwrap_or(50).min(256);
    let temperature = req.temperature.unwrap_or(0.8);
    let top_p = req.top_p.unwrap_or(0.9);

    let prompt_ids = tokenize_prompt(&state, &req.prompt)?;
    let start = Instant::now();

    let gen = state.generator.read().await;
    let output_ids = if temperature < 1e-6 {
        gen.generate_greedy(&prompt_ids, prompt_ids.len() + max_tokens)
    } else {
        gen.generate_top_p(
            &prompt_ids,
            prompt_ids.len() + max_tokens,
            top_p,
            temperature,
            None,
            1.0,
        )
    }
    .map_err(ApiError::generation)?;

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let completion_tokens = output_ids.len().saturating_sub(prompt_ids.len());
    let total_tokens = output_ids.len();
    if let Some(t) = &state.telemetry {
        t.record_request(elapsed_ms, completion_tokens);
    }
    let text = decode_tokens(&state, &output_ids[prompt_ids.len()..]).unwrap_or_else(|_| {
        output_ids
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    });

    Ok(Json(GenerationResponse {
        text,
        tokens: output_ids,
        finish_reason: "length".to_string(),
        usage: UsageStats {
            prompt_tokens: prompt_ids.len(),
            completion_tokens,
            total_tokens,
            time_ms: elapsed_ms,
        },
    }))
}

async fn chat_completions(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, ApiError> {
    let last = req
        .messages
        .last()
        .ok_or(ApiError::bad_request("messages empty".into()))?;
    let prompt = format!("{}: {}", last.role, last.content);
    let max_tokens = req.max_tokens.unwrap_or(100).min(256);
    let temperature = req.temperature.unwrap_or(0.8);

    let prompt_ids = tokenize_prompt(&state, &prompt)?;
    let start = Instant::now();

    let gen = state.generator.read().await;
    let output_ids = gen
        .generate_top_p(
            &prompt_ids,
            prompt_ids.len() + max_tokens,
            0.9,
            temperature,
            None,
            1.0,
        )
        .map_err(ApiError::generation)?;

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let completion_tokens = output_ids.len().saturating_sub(prompt_ids.len());
    if let Some(t) = &state.telemetry {
        t.record_request(elapsed_ms, completion_tokens);
    }
    let text = decode_tokens(&state, &output_ids[prompt_ids.len()..]).unwrap_or_else(|_| "".into());

    Ok(Json(ChatCompletionResponse {
        choices: vec![ChatChoice {
            message: ChatMessage {
                role: "assistant".to_string(),
                content: text,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: UsageStats {
            prompt_tokens: prompt_ids.len(),
            completion_tokens,
            total_tokens: output_ids.len(),
            time_ms: elapsed_ms,
        },
    }))
}

async fn list_models(State(_state): State<Arc<ServerState>>) -> Json<ModelsListResponse> {
    Json(ModelsListResponse {
        data: vec![ModelInfo {
            id: "bitnet-oxidized".to_string(),
            object: "model".to_string(),
            owned_by: "bitnet-oxidized".to_string(),
        }],
    })
}

async fn metrics(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
    if let Some(t) = &state.telemetry {
        let body = t.export_metrics();
        ([(header::CONTENT_TYPE, "text/plain; charset=utf-8")], body).into_response()
    } else {
        Json(MetricsResponse {
            requests_total: 0,
            tokens_generated_total: 0,
        })
        .into_response()
    }
}

fn tokenize_prompt(state: &ServerState, prompt: &str) -> Result<Vec<usize>, ApiError> {
    if let Some(tok) = &state.tokenizer {
        Ok(tok.encode(prompt).map_err(ApiError::tokenizer)?)
    } else {
        Ok(simple_tokenizer(prompt))
    }
}

fn decode_tokens(state: &ServerState, ids: &[usize]) -> Result<String, BitNetError> {
    if let Some(tok) = &state.tokenizer {
        tok.decode(ids)
    } else {
        Ok(ids
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(" "))
    }
}

fn simple_tokenizer(prompt: &str) -> Vec<usize> {
    const VOCAB: usize = 256;
    prompt
        .split_whitespace()
        .map(|s| {
            let h = s.bytes().fold(0u64, |a, b| a.wrapping_add(b as u64));
            (h as usize) % VOCAB
        })
        .collect()
}

pub struct ApiError(Box<dyn std::error::Error + Send + Sync>);

impl ApiError {
    fn bad_request(msg: String) -> Self {
        ApiError(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            msg,
        )))
    }
    fn generation(e: anyhow::Error) -> Self {
        ApiError(e.into())
    }
    fn tokenizer(e: BitNetError) -> Self {
        ApiError(e.into())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        (StatusCode::INTERNAL_SERVER_ERROR, self.0.to_string()).into_response()
    }
}
