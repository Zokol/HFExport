//! Concurrent, resumable file downloader with progress reporting.
//!
//! The main entry points are:
//!
//! - [`download_file`] – downloads a single URL to a local path, resuming if a
//!   partial file already exists and the server supports `Range` requests.
//! - [`download_model`] – fetches the Hub file list for a repository and
//!   dispatches one [`download_file`] call per file.
//!
//! Progress is reported through [`indicatif::MultiProgress`] so that the caller
//! controls the overall terminal output, even when multiple downloads run
//! concurrently.

use std::path::{Path, PathBuf};

use anyhow::Context;
use futures_util::StreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use reqwest::{Client, StatusCode};
use tokio::{
    fs::{self, OpenOptions},
    io::AsyncWriteExt,
};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single file to be downloaded.
///
/// Construct one per file and pass it to [`download_file`].
pub struct DownloadTask {
    /// The full HTTPS URL of the resource (used as the LFS / HTTP fallback).
    pub url: String,

    /// Absolute path where the downloaded bytes should be written.
    ///
    /// Parent directories are created automatically by [`download_file`].
    pub dest: PathBuf,

    /// Total expected size in bytes, used to size the progress bar.
    ///
    /// If `None` the bar is displayed in "unknown length" spinner mode.
    pub expected_size: Option<u64>,

    // ------------------------------------------------------------------
    // Fields required for the Xet fast-path.
    // ------------------------------------------------------------------

    /// Repository identifier in `"{owner}/{model}"` format
    /// (e.g. `"microsoft/phi-2"`).
    pub repo_id: String,

    /// Path of the file inside the repository
    /// (e.g. `"model.safetensors"` or `"subfolder/weights.bin"`).
    pub filename_in_repo: String,

    /// Git revision to resolve against (e.g. `"main"`).
    pub revision: String,

    /// Repository type: `"model"`, `"dataset"`, or `"space"`.
    ///
    /// Passed to the Xet token endpoint to build the correct API path.
    pub repo_type: String,
}

// ---------------------------------------------------------------------------
// Progress bar helpers
// ---------------------------------------------------------------------------

/// The template string used for every per-file progress bar.
///
/// Layout: `⠋ filename.bin [████████████--------] 512 MiB/1.2 GiB (98 MiB/s, 7s)`
const PROGRESS_TEMPLATE: &str =
    "{spinner} {wide_msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})";

/// Spinner tick characters (braille sequence — looks smooth at 100 ms/tick).
const SPINNER_CHARS: &str = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏";

/// Build a styled [`ProgressBar`] ready for download reporting.
///
/// - If `total` is `Some`, the bar shows a bounded progress indicator.
/// - If `total` is `None`, the bar runs in spinner-only mode (no ETA).
fn make_progress_bar(total: Option<u64>) -> ProgressBar {
    let pb = match total {
        Some(len) => ProgressBar::new(len),
        None => ProgressBar::new_spinner(),
    };

    let style = ProgressStyle::with_template(PROGRESS_TEMPLATE)
        // The template is a compile-time constant, so a panic here would be a
        // programming error rather than a runtime condition.
        .expect("progress bar template is valid")
        .tick_chars(SPINNER_CHARS)
        .progress_chars("█▓░");

    pb.set_style(style);
    pb
}

// ---------------------------------------------------------------------------
// Core download function
// ---------------------------------------------------------------------------

/// Download a single file described by `task`.
///
/// # Resume behaviour
///
/// If `task.dest` already exists on disk its current size is determined via
/// [`tokio::fs::metadata`]. The request is sent with
/// `Range: bytes={existing_size}-` so that only the missing tail is
/// transferred.
///
/// - If the server responds with **416 Range Not Satisfiable** the file is
///   already complete and the function returns `Ok(())` immediately.
/// - A **206 Partial Content** response is expected for a resumed download.
/// - A **200 OK** response is expected for a fresh download (server ignored
///   the `Range` header or the file did not exist yet).
/// - Any other status is treated as an error.
///
/// # Errors
///
/// Returns an error for network failures, unexpected HTTP status codes, or
/// I/O errors while writing to disk.
pub async fn download_file(
    client: &Client,
    task: &DownloadTask,
    token: Option<&str>,
    multi_progress: &MultiProgress,
) -> anyhow::Result<()> {
    // ------------------------------------------------------------------
    // 1. Set up the progress bar up-front so it can be passed to both the
    //    Xet fast-path and the HTTP fallback.
    // ------------------------------------------------------------------
    let pb = make_progress_bar(task.expected_size);

    // Show the filename (not the full URL) as the progress bar message.
    let display_name = task
        .dest
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| task.url.clone());
    pb.set_message(display_name.clone());

    // Register with the multi-progress renderer so output is coordinated.
    let pb = multi_progress.add(pb);

    // ------------------------------------------------------------------
    // 2. Try the Xet fast-path first.
    //
    //    `download_file_xet` returns an error both for genuine protocol
    //    failures and for files that are not stored on Xet ("File not on
    //    Xet storage").  In either case we log the reason and fall through
    //    to the standard HTTP range-based LFS download below.
    // ------------------------------------------------------------------
    match crate::xet::download_file_xet(
        client,
        &task.repo_id,
        &task.repo_type,
        &task.filename_in_repo,
        &task.revision,
        &task.dest,
        token,
        &pb,
    )
    .await
    {
        Ok(()) => {
            pb.finish_with_message(format!("{display_name} (done)"));
            return Ok(());
        }
        Err(e) => {
            // Log at a low level and fall through to the LFS download.
            pb.set_message(format!("{display_name} (xet unavailable: {e:#}, using LFS)"));
        }
    }

    // ------------------------------------------------------------------
    // 3. Check for an existing partial file so we can resume the HTTP
    //    range-based download.
    // ------------------------------------------------------------------
    let resume_offset: u64 = match fs::metadata(&task.dest).await {
        Ok(meta) => meta.len(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => 0,
        Err(e) => {
            return Err(e).with_context(|| {
                format!(
                    "failed to stat destination file `{}`",
                    task.dest.display()
                )
            });
        }
    };

    // ------------------------------------------------------------------
    // 4. Build the HTTP request, adding optional auth and Range headers.
    // ------------------------------------------------------------------
    let mut request = client.get(&task.url);

    if let Some(tok) = token {
        request = request.bearer_auth(tok);
    }

    if resume_offset > 0 {
        request = request.header(
            reqwest::header::RANGE,
            format!("bytes={resume_offset}-"),
        );
    }

    // ------------------------------------------------------------------
    // 5. Send and inspect the status code.
    // ------------------------------------------------------------------
    let response = request
        .send()
        .await
        .with_context(|| format!("HTTP GET `{}` failed", task.url))?;

    let status = response.status();

    // 416 means the server thinks we already have everything.
    if status == StatusCode::RANGE_NOT_SATISFIABLE {
        pb.finish_with_message(format!("{display_name} (already complete)"));
        return Ok(());
    }

    // Any status other than 200 OK or 206 Partial Content is unexpected.
    if status != StatusCode::OK && status != StatusCode::PARTIAL_CONTENT {
        anyhow::bail!(
            "unexpected HTTP {status} from `{}`",
            task.url
        );
    }

    // ------------------------------------------------------------------
    // 6. Create parent directories and open (or reopen) the destination.
    // ------------------------------------------------------------------
    if let Some(parent) = task.dest.parent() {
        fs::create_dir_all(parent).await.with_context(|| {
            format!("failed to create directory `{}`", parent.display())
        })?;
    }

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&task.dest)
        .await
        .with_context(|| {
            format!("failed to open `{}` for writing", task.dest.display())
        })?;

    // Reset the progress bar label now that we are doing an LFS transfer.
    pb.set_message(display_name.clone());

    // Advance the bar to the already-downloaded position so the display
    // reflects what is actually on disk from the start.
    pb.set_position(resume_offset);

    // ------------------------------------------------------------------
    // 7. Stream the response body to disk.
    // ------------------------------------------------------------------
    let mut stream = response.bytes_stream();

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result
            .with_context(|| format!("stream error while downloading `{}`", task.url))?;

        file.write_all(&chunk)
            .await
            .with_context(|| {
                format!("I/O error writing to `{}`", task.dest.display())
            })?;

        // Advance by the number of bytes just written.
        pb.inc(chunk.len() as u64);
    }

    // Flush any OS-level buffers before we declare success.
    file.flush()
        .await
        .with_context(|| format!("failed to flush `{}`", task.dest.display()))?;

    pb.finish_with_message(format!("{display_name} (done)"));

    Ok(())
}

// ---------------------------------------------------------------------------
// Model-level helper
// ---------------------------------------------------------------------------

/// Download every file in a HuggingFace model repository.
///
/// `repo_id` must be in `"{owner}/{model}"` format, matching the HuggingFace
/// Hub convention (e.g. `"microsoft/phi-2"`).
///
/// Files are written under:
/// ```text
/// {output_dir}/{owner}/{model}/{rfilename}
/// ```
///
/// # Errors
///
/// - `repo_id` does not contain exactly one `'/'`.
/// - The Hub API call to [`crate::api::HfClient::get_model_info`] fails.
/// - Any individual file download fails (the first error aborts the function).
pub async fn download_model(
    hf_client: &crate::api::HfClient,
    reqwest_client: &Client,
    repo_id: &str,
    output_dir: &Path,
    token: Option<&str>,
    multi_progress: &MultiProgress,
) -> anyhow::Result<()> {
    // ------------------------------------------------------------------
    // 1. Resolve the owner / model name from the repo_id.
    // ------------------------------------------------------------------
    let (owner, model_name) = repo_id.split_once('/').with_context(|| {
        format!(
            "invalid repo_id `{repo_id}`: expected `owner/model` format (e.g. `microsoft/phi-2`)"
        )
    })?;

    // ------------------------------------------------------------------
    // 2. Fetch the file list from the Hub API.
    // ------------------------------------------------------------------
    let model_info = hf_client
        .get_model_info(repo_id)
        .await
        .with_context(|| format!("failed to fetch model info for `{repo_id}`"))?;

    let file_count = model_info.siblings.len();
    println!(
        "Downloading {file_count} file(s) from `{repo_id}` into `{}`",
        output_dir.join(owner).join(model_name).display()
    );

    // ------------------------------------------------------------------
    // 3. Build the model output directory: {output_dir}/{owner}/{model}.
    // ------------------------------------------------------------------
    let model_dir = output_dir.join(owner).join(model_name);

    // ------------------------------------------------------------------
    // 4. Download each file sequentially.
    //    (Callers who want concurrency can spawn tasks around this fn.)
    // ------------------------------------------------------------------
    let mut downloaded = 0usize;
    let mut skipped = 0usize;

    for sibling in &model_info.siblings {
        // Preserve any sub-directory structure inside the repo
        // (e.g. `subfolder/weights.bin`).
        let dest = model_dir.join(&sibling.rfilename);

        let download_url = hf_client.download_url(repo_id, &sibling.rfilename);

        let task = DownloadTask {
            url: download_url,
            dest: dest.clone(),
            expected_size: sibling.size,
            repo_id: repo_id.to_string(),
            filename_in_repo: sibling.rfilename.clone(),
            revision: "main".to_string(),
            repo_type: "model".to_string(),
        };

        // Check whether the file is already fully present before dispatching
        // the download.  download_file itself handles 416, but this short-
        // circuit avoids even opening a network connection for complete files.
        if let Some(expected) = sibling.size
            && let Ok(meta) = fs::metadata(&dest).await
            && meta.len() == expected
        {
            println!("  skipping `{}` (already complete)", sibling.rfilename);
            skipped += 1;
            continue;
        }

        download_file(reqwest_client, &task, token, multi_progress)
            .await
            .with_context(|| {
                format!(
                    "failed to download `{}` from `{}`",
                    sibling.rfilename, repo_id
                )
            })?;

        downloaded += 1;
    }

    // ------------------------------------------------------------------
    // 5. Print a summary.
    // ------------------------------------------------------------------
    println!(
        "\nDone. {downloaded} file(s) downloaded, {skipped} skipped (already complete)."
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the progress bar template compiles without panicking.
    #[test]
    fn progress_bar_template_is_valid() {
        let _ = make_progress_bar(Some(1024));
        let _ = make_progress_bar(None);
    }

    /// Verify that DownloadTask fields are accessible (basic struct construction).
    #[test]
    fn download_task_fields() {
        let task = DownloadTask {
            url: "https://example.com/file.bin".to_owned(),
            dest: PathBuf::from("/tmp/file.bin"),
            expected_size: Some(1024),
            repo_id: "owner/model".to_owned(),
            filename_in_repo: "file.bin".to_owned(),
            revision: "main".to_owned(),
            repo_type: "model".to_owned(),
        };
        assert_eq!(task.url, "https://example.com/file.bin");
        assert_eq!(task.dest, PathBuf::from("/tmp/file.bin"));
        assert_eq!(task.expected_size, Some(1024));
        assert_eq!(task.repo_id, "owner/model");
        assert_eq!(task.filename_in_repo, "file.bin");
        assert_eq!(task.revision, "main");
        assert_eq!(task.repo_type, "model");
    }
}
