mod api;
mod config;
mod downloader;
mod xet;

use config::{Commands, parse_cli, save_token};
use indicatif::MultiProgress;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let (cli, config) = parse_cli()?;

    match cli.command {
        // ----------------------------------------------------------------
        // hfexport login
        // ----------------------------------------------------------------
        Commands::Login => {
            // rpassword reads from /dev/tty directly, so the token is never
            // echoed even when stdout is piped.
            let token = rpassword::prompt_password(
                "Paste your HuggingFace token (input hidden): ",
            )
            .map_err(|e| anyhow::anyhow!("failed to read token: {e}"))?;

            let token = token.trim().to_owned();
            if token.is_empty() {
                eprintln!("error: token must not be empty");
                std::process::exit(1);
            }

            save_token(&token)?;
            println!("Token saved successfully.");
        }

        // ----------------------------------------------------------------
        // hfexport download -m <repo> [-m <repo>...] [-o <dir>]
        // ----------------------------------------------------------------
        Commands::Download { .. } => {
            // clap enforces `required = true` on --models, but guard anyway.
            if config.models.is_empty() {
                eprintln!("error: at least one --models argument is required");
                std::process::exit(1);
            }

            // Announce authentication status before any network I/O.
            match &config.token {
                Some(_) => println!("Authenticated download (token loaded from cache)."),
                None => println!(
                    "No token found. Only public repositories can be downloaded.\n\
                     Run `hfexport login` to store your HuggingFace API token."
                ),
            }

            println!("Output directory : {}", config.output_dir.display());
            println!("Models to download ({}):", config.models.len());
            for repo in &config.models {
                println!("  - {repo}");
            }

            // ------------------------------------------------------------------
            // Build shared infrastructure once, then iterate over models.
            //
            // HfClient owns its own internal reqwest::Client (used for API
            // metadata calls).  We build a *second* plain Client for the actual
            // binary downloads in download_file so that each has its own
            // connection pool and auth headers are only added per-request by
            // the downloader (not baked into default headers).
            // ------------------------------------------------------------------
            let hf_client = api::HfClient::new(config.token.clone())
                .map_err(|e| anyhow::anyhow!("failed to build HF API client: {e}"))?;

            let reqwest_client = reqwest::Client::builder()
                .build()
                .map_err(|e| anyhow::anyhow!("failed to build HTTP client: {e}"))?;

            let multi_progress = MultiProgress::new();

            let token_ref: Option<&str> = config.token.as_deref();

            let mut any_failed = false;

            for repo_id in &config.models {
                if let Err(e) = downloader::download_model(
                    &hf_client,
                    &reqwest_client,
                    repo_id,
                    &config.output_dir,
                    token_ref,
                    &multi_progress,
                )
                .await
                {
                    eprintln!("error: failed to download `{repo_id}`: {e:#}");
                    any_failed = true;
                }
            }

            if any_failed {
                std::process::exit(1);
            }
        }
    }

    Ok(())
}
