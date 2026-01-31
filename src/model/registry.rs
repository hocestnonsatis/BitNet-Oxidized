//! Model zoo / registry: named models and from_pretrained loading.
//!
//! - Built-in: `demo` → in-memory demo model.
//! - Path: any string that is an existing file path → load via GGUF.
//! - Registry: optional named entries (e.g. from env or config file).

use crate::errors::BitNetError;
use crate::model::{create_demo_model, load_gguf, BitNetModel};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Single model entry in the registry (name → path or demo).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Display name / id (e.g. "bitnet-1.58-large").
    pub name: String,
    /// Path to GGUF file. If None, name "demo" is treated as in-memory demo.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<PathBuf>,
    /// Optional short description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Registry of named models. Use with [from_pretrained].
#[derive(Debug, Clone, Default)]
pub struct ModelRegistry {
    entries: HashMap<String, ModelEntry>,
}

impl ModelRegistry {
    /// New empty registry (only path-based and "demo" will work).
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Default registry with built-in "demo" entry (no path = in-memory).
    pub fn default_registry() -> Self {
        let mut r = Self::new();
        r.register(ModelEntry {
            name: "demo".to_string(),
            path: None,
            description: Some("In-memory demo model (random small BitNet)".to_string()),
        });
        r
    }

    /// Register an entry by name. Overwrites if name exists.
    pub fn register(&mut self, entry: ModelEntry) {
        self.entries.insert(entry.name.clone(), entry);
    }

    /// Register a model by name and path.
    pub fn register_path(&mut self, name: impl Into<String>, path: impl AsRef<Path>) {
        let name = name.into();
        self.entries.insert(
            name.clone(),
            ModelEntry {
                name,
                path: Some(path.as_ref().to_path_buf()),
                description: None,
            },
        );
    }

    /// Get entry by name.
    pub fn get(&self, name: &str) -> Option<&ModelEntry> {
        self.entries.get(name)
    }

    /// List all registered names.
    pub fn names(&self) -> Vec<String> {
        self.entries.keys().cloned().collect()
    }

    /// Load from env: BITNET_REGISTRY=/path/to/registry.json, or BITNET_MODEL_<NAME>=/path/to/model.gguf.
    /// Merges with existing entries (env wins). Does not fail if file missing.
    pub fn load_from_env(&mut self) {
        if let Ok(path) = std::env::var("BITNET_REGISTRY") {
            let p = PathBuf::from(path);
            if p.exists() {
                if let Ok(content) = std::fs::read_to_string(&p) {
                    if let Ok(entries) = serde_json::from_str::<Vec<ModelEntry>>(&content) {
                        for e in entries {
                            self.register(e);
                        }
                    }
                }
            }
        }
        for (key, value) in std::env::vars() {
            if key.starts_with("BITNET_MODEL_") && !value.is_empty() {
                let name = key
                    .trim_start_matches("BITNET_MODEL_")
                    .to_lowercase()
                    .replace('_', "-");
                let path = PathBuf::from(value);
                if path.exists() {
                    self.register_path(name, path);
                }
            }
        }
    }
}

/// Load a model by name or path.
///
/// - `"demo"` → in-memory demo model (no file).
/// - Existing file path (e.g. `"/path/to/model.gguf"`) → [load_gguf].
/// - Otherwise, look up name in `registry` and load its path.
pub fn from_pretrained(
    name_or_path: impl AsRef<str>,
    registry: &ModelRegistry,
) -> Result<BitNetModel, BitNetError> {
    let s = name_or_path.as_ref().trim();
    if s.is_empty() {
        return Err(BitNetError::InvalidInput(
            "model name or path must not be empty".to_string(),
        ));
    }

    if s.eq_ignore_ascii_case("demo") {
        return Ok(create_demo_model());
    }

    let path = Path::new(s);
    if path.exists() && path.is_file() {
        return load_gguf(path);
    }

    if let Some(entry) = registry.get(s) {
        if let Some(ref p) = entry.path {
            if p.exists() {
                return load_gguf(p);
            }
            return Err(BitNetError::InvalidInput(format!(
                "registry model '{}' path does not exist: {}",
                s,
                p.display()
            )));
        }
        if entry.name.eq_ignore_ascii_case("demo") {
            return Ok(create_demo_model());
        }
    }

    Err(BitNetError::InvalidInput(
        format!(
            "unknown model name or path: '{}'. Use 'demo', a path to a GGUF file, or register the model.",
            s
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_pretrained_demo() {
        let reg = ModelRegistry::default_registry();
        let model = from_pretrained("demo", &reg).unwrap();
        assert_eq!(model.config.vocab_size, 256);
    }

    #[test]
    fn from_pretrained_demo_case_insensitive() {
        let reg = ModelRegistry::default_registry();
        let _ = from_pretrained("DEMO", &reg).unwrap();
    }

    #[test]
    fn from_pretrained_unknown_fails() {
        let reg = ModelRegistry::default_registry();
        let r = from_pretrained("nonexistent-model-xyz", &reg);
        assert!(r.is_err());
    }
}
