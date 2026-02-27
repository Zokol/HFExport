//! Configuration types and CLI argument parsing for HFExport.
//!
//! This module defines the [`Cli`] struct parsed by `clap`, the [`Commands`]
//! subcommand enum, token persistence helpers (`save_token` / `load_token`),
//! and the [`Config`] struct that collects everything `main` needs to run a
//! download job.
//!
//! # Token storage
//!
//! Tokens are stored in `~/.cache/hfexport/token` (i.e. outside the project
//! directory) with mode `0o600` on Unix so only the current user can read them.

use std::path::PathBuf;

use anyhow::Context;
use clap::{Parser, Subcommand};

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

/// Download HuggingFace models from the command line.
#[derive(Debug, Parser)]
#[command(
    name = "hfexport",
    version,
    about = "Download HuggingFace models to a local directory",
    long_about = None,
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

/// Available subcommands.
#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Save a HuggingFace API token to disk for authenticated downloads.
    ///
    /// After running `hfexport login` you will be prompted to paste your token.
    /// The token is stored in `~/.cache/hfexport/token` with mode 0o600.
    Login,

    /// Download one or more HuggingFace model repositories.
    ///
    /// Example:
    ///   hfexport download -m microsoft/phi-2 -m mistralai/Mistral-7B-v0.1
    Download {
        /// One or more HuggingFace repo IDs to download (e.g. `microsoft/phi-2`).
        #[arg(short = 'm', long = "models", required = true, num_args = 1..)]
        models: Vec<String>,

        /// Directory where model files are saved.
        #[arg(short = 'o', long = "output", default_value = "./models")]
        output: PathBuf,
    },
}

// ---------------------------------------------------------------------------
// Token storage
// ---------------------------------------------------------------------------

/// Returns the canonical path to the stored token file.
///
/// On Unix this resolves to `~/.cache/hfexport/token`.
/// The function panics only when `dirs::cache_dir()` returns `None`, which
/// happens exclusively in headless environments with no `$HOME`; that case is
/// not recoverable in a CLI tool.
fn token_path() -> PathBuf {
    dirs::cache_dir()
        .expect("could not determine cache directory (is $HOME set?)")
        .join("hfexport")
        .join("token")
}

/// Persist `token` to `~/.cache/hfexport/token`.
///
/// The parent directory is created if it does not exist. On Unix the file is
/// written with mode `0o600` so that only the owning user can read it.
///
/// # Errors
///
/// Returns an error if the directory cannot be created or the file cannot be
/// written.
pub fn save_token(token: &str) -> anyhow::Result<()> {
    let path = token_path();

    // Create ~/.cache/hfexport/ if it doesn't exist yet.
    let parent = path
        .parent()
        .expect("token path always has a parent directory");
    std::fs::create_dir_all(parent)
        .with_context(|| format!("failed to create directory {}", parent.display()))?;

    std::fs::write(&path, token)
        .with_context(|| format!("failed to write token to {}", path.display()))?;

    // Restrict permissions to owner-only read/write on Unix.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o600);
        std::fs::set_permissions(&path, perms)
            .with_context(|| format!("failed to set permissions on {}", path.display()))?;
    }

    Ok(())
}

/// Read the stored token from `~/.cache/hfexport/token`.
///
/// Returns `None` if the file does not exist or cannot be read (e.g. first
/// run before `hfexport login`).
pub fn load_token() -> Option<String> {
    let path = token_path();
    std::fs::read_to_string(&path)
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
}

// ---------------------------------------------------------------------------
// Derived application config
// ---------------------------------------------------------------------------

/// Application-level configuration derived from the parsed CLI arguments.
///
/// This struct is only fully populated for the `download` subcommand; for
/// `login` the `models` vec and `output_dir` carry their defaults and are
/// ignored by `main`.
pub struct Config {
    /// HuggingFace API token loaded from disk (may be absent for public repos).
    pub token: Option<String>,
    /// Directory into which model files will be written.
    pub output_dir: PathBuf,
    /// List of HuggingFace repo IDs requested by the user.
    pub models: Vec<String>,
}

/// Parse command-line arguments and build the application [`Config`].
///
/// This is the single entry point called by `main`. It:
/// 1. Runs `clap` argument parsing (exits with usage on error).
/// 2. Loads any saved token from disk.
/// 3. Constructs a [`Config`] from the parsed arguments.
///
/// Returns both the raw [`Cli`] (so `main` can `match` on the subcommand) and
/// the derived [`Config`].
pub fn parse_cli() -> anyhow::Result<(Cli, Config)> {
    let cli = Cli::parse();

    let token = load_token();

    let (output_dir, models) = match &cli.command {
        Commands::Login => (PathBuf::from("./models"), Vec::new()),
        Commands::Download { output, models } => (output.clone(), models.clone()),
    };

    let config = Config {
        token,
        output_dir,
        models,
    };

    Ok((cli, config))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_path_is_outside_project() {
        let path = token_path();
        // The token path must live under the system cache dir, not the cwd.
        // We verify it is absolute and contains "hfexport" and "token".
        assert!(path.is_absolute(), "token path must be absolute: {path:?}");
        assert_eq!(path.file_name().unwrap(), "token");
        assert_eq!(
            path.parent().unwrap().file_name().unwrap(),
            "hfexport"
        );
    }

    #[test]
    fn save_and_load_token_roundtrip() {
        // Write a test token and read it back.
        let test_token = "hf_test_ROUNDTRIP_TOKEN_12345";
        save_token(test_token).expect("save_token failed");

        let loaded = load_token().expect("load_token returned None after save");
        assert_eq!(loaded, test_token);
    }

    #[test]
    fn load_token_returns_none_for_empty_file() {
        // If the file exists but is whitespace-only, load_token should treat
        // it as absent.
        let path = token_path();
        // Best-effort: only run if we can create the directory.
        if std::fs::create_dir_all(path.parent().unwrap()).is_ok() {
            std::fs::write(&path, "   \n").unwrap();
            assert!(
                load_token().is_none(),
                "expected None for whitespace-only token file"
            );
            // Clean up so the roundtrip test isn't affected.
            let _ = std::fs::remove_file(&path);
        }
    }
}
