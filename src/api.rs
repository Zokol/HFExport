//! HuggingFace Hub API client.
//!
//! This module implements communication with the HuggingFace Hub REST API,
//! including fetching repository metadata (file listing, sizes) and resolving
//! download URLs for individual model files.
//!
//! # Authentication
//!
//! Pass a HuggingFace API token (obtained from <https://huggingface.co/settings/tokens>)
//! to [`HfClient::new`] to enable authenticated requests. Authenticated requests
//! are required for gated models and increase rate limits.
//!
//! # Example
//!
//! ```no_run
//! use HFExport::api::HfClient;
//!
//! # #[tokio::main]
//! # async fn main() -> anyhow::Result<()> {
//! let client = HfClient::new(None)?;
//! let info = client.get_model_info("microsoft/phi-2").await?;
//! println!("model {} has {} files", info.id, info.siblings.len());
//! # Ok(())
//! # }
//! ```

use anyhow::Context;
use reqwest::{
    header::{HeaderMap, HeaderValue, USER_AGENT},
    Client, StatusCode,
};
use serde::Deserialize;

// ---------------------------------------------------------------------------
// API response types
// ---------------------------------------------------------------------------

/// Metadata for a single file within a HuggingFace model repository.
///
/// This mirrors the objects returned in the `siblings` array of the
/// `/api/models/{repo_id}` endpoint.
#[derive(Debug, Deserialize)]
pub struct ModelFile {
    /// Relative filename inside the repository (e.g. `"config.json"`).
    pub rfilename: String,

    /// File size in bytes, if reported by the API.
    ///
    /// The Hub omits this field for some repository types, hence `Option`.
    pub size: Option<u64>,
}

/// Top-level metadata for a HuggingFace model repository.
///
/// Returned by [`HfClient::get_model_info`]. Only the fields needed by
/// HFExport are deserialised; the Hub response contains many more fields that
/// are silently ignored via `#[serde(deny_unknown_fields)]` being absent.
#[derive(Debug, Deserialize)]
pub struct ModelInfo {
    /// Full repo identifier in `{owner}/{name}` form (e.g. `"microsoft/phi-2"`).
    // `id` is part of the public API surface and used in tests; the binary
    // does not inspect it directly, so we suppress the dead_code lint here.
    #[allow(dead_code)]
    pub id: String,

    /// Every file present in the repository's default revision.
    pub siblings: Vec<ModelFile>,
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// Authenticated (or anonymous) HuggingFace Hub HTTP client.
///
/// Construct once via [`HfClient::new`] and reuse across requests — the inner
/// [`reqwest::Client`] maintains a connection pool.
pub struct HfClient {
    client: Client,
    token: Option<String>,
}

impl HfClient {
    /// Build a new client.
    ///
    /// `token` should be a HuggingFace API token (`hf_…`). Pass `None` for
    /// anonymous access (public repositories only, lower rate limits).
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying [`reqwest::Client`] cannot be
    /// constructed (e.g. the TLS backend fails to initialise).
    pub fn new(token: Option<String>) -> anyhow::Result<Self> {
        let mut default_headers = HeaderMap::new();

        // Every request carries this User-Agent so the Hub can identify traffic
        // from this tool. HeaderValue::from_static is infallible for string literals.
        default_headers.insert(USER_AGENT, HeaderValue::from_static("HFExport/0.1"));

        let client = Client::builder()
            .default_headers(default_headers)
            .build()
            .context("failed to build HTTP client")?;

        Ok(Self { client, token })
    }

    /// Fetch repository metadata from the HuggingFace Hub API.
    ///
    /// Calls `GET https://huggingface.co/api/models/{repo_id}` and deserialises
    /// the JSON response into a [`ModelInfo`].
    ///
    /// # Errors
    ///
    /// - Network or TLS errors from [`reqwest`].
    /// - Non-2xx HTTP status (e.g. 401 Unauthorised, 404 Not Found).
    /// - JSON deserialisation failure if the Hub response schema changes.
    pub async fn get_model_info(&self, repo_id: &str) -> anyhow::Result<ModelInfo> {
        let url = format!("https://huggingface.co/api/models/{repo_id}");

        let mut request = self.client.get(&url);

        // Attach bearer token only when one was supplied; anonymous requests
        // omit the Authorization header entirely.
        if let Some(token) = &self.token {
            request = request.bearer_auth(token);
        }

        let response = request
            .send()
            .await
            .with_context(|| format!("HTTP request to {url} failed"))?;

        // Translate non-2xx status codes into descriptive errors before
        // attempting to deserialise the body.
        let status = response.status();
        if !status.is_success() {
            let detail = match status {
                StatusCode::UNAUTHORIZED => {
                    " (token missing or invalid — run `hfexport login`)".to_owned()
                }
                StatusCode::NOT_FOUND => format!(" (repository `{repo_id}` not found)"),
                StatusCode::FORBIDDEN => {
                    format!(" (access to `{repo_id}` is gated — request access on the Hub)")
                }
                _ => String::new(),
            };
            anyhow::bail!("Hub API returned {status} for {url}{detail}");
        }

        let info = response
            .json::<ModelInfo>()
            .await
            .with_context(|| format!("failed to deserialise model info for `{repo_id}`"))?;

        Ok(info)
    }

    /// Build the HTTPS URL used to download a single file from a repository.
    ///
    /// The returned URL resolves the file from the `main` branch (default
    /// revision). Pass it to the downloader to fetch the actual bytes.
    ///
    /// This function is infallible — it only performs string formatting.
    ///
    /// # Example
    ///
    /// ```
    /// # use HFExport::api::HfClient;
    /// let client = HfClient::new(None).unwrap();
    /// let url = client.download_url("microsoft/phi-2", "config.json");
    /// assert_eq!(url, "https://huggingface.co/microsoft/phi-2/resolve/main/config.json");
    /// ```
    pub fn download_url(&self, repo_id: &str, filename: &str) -> String {
        format!("https://huggingface.co/{repo_id}/resolve/main/{filename}")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_succeeds_without_token() {
        HfClient::new(None).expect("client construction should succeed without a token");
    }

    #[test]
    fn new_succeeds_with_token() {
        HfClient::new(Some("hf_test_token".to_owned()))
            .expect("client construction should succeed with a token");
    }

    #[test]
    fn download_url_format() {
        let client = HfClient::new(None).unwrap();

        assert_eq!(
            client.download_url("microsoft/phi-2", "config.json"),
            "https://huggingface.co/microsoft/phi-2/resolve/main/config.json"
        );

        // Nested path inside the repo.
        assert_eq!(
            client.download_url("mistralai/Mistral-7B-v0.1", "model-00001-of-00002.safetensors"),
            "https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/model-00001-of-00002.safetensors"
        );
    }

    #[test]
    fn model_file_deserialises_with_size() {
        let json = r#"{"rfilename": "config.json", "size": 1234}"#;
        let file: ModelFile = serde_json::from_str(json).unwrap();
        assert_eq!(file.rfilename, "config.json");
        assert_eq!(file.size, Some(1234));
    }

    #[test]
    fn model_file_deserialises_without_size() {
        let json = r#"{"rfilename": "README.md"}"#;
        let file: ModelFile = serde_json::from_str(json).unwrap();
        assert_eq!(file.rfilename, "README.md");
        assert!(file.size.is_none());
    }

    #[test]
    fn model_info_deserialises_full_payload() {
        let json = r#"{
            "id": "microsoft/phi-2",
            "siblings": [
                {"rfilename": "config.json", "size": 1234},
                {"rfilename": "model.safetensors", "size": 5368709120}
            ]
        }"#;

        let info: ModelInfo = serde_json::from_str(json).unwrap();
        assert_eq!(info.id, "microsoft/phi-2");
        assert_eq!(info.siblings.len(), 2);
        assert_eq!(info.siblings[0].rfilename, "config.json");
        assert_eq!(info.siblings[0].size, Some(1234));
        assert_eq!(info.siblings[1].rfilename, "model.safetensors");
        assert_eq!(info.siblings[1].size, Some(5368709120));
    }

    #[test]
    fn model_info_tolerates_extra_json_fields() {
        // The Hub API returns many more fields; they must be silently ignored.
        let json = r#"{
            "id": "owner/model",
            "private": false,
            "lastModified": "2024-01-01T00:00:00.000Z",
            "siblings": [{"rfilename": "file.bin", "size": 999}],
            "tags": ["transformers", "pytorch"]
        }"#;

        let info: ModelInfo = serde_json::from_str(json).unwrap();
        assert_eq!(info.id, "owner/model");
        assert_eq!(info.siblings.len(), 1);
    }
}
