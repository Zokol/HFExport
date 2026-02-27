//! Xet protocol download module.
//!
//! HuggingFace stores large model files on Xet storage, a content-addressed
//! blob store backed by a chunk reconstruction protocol.  This module
//! implements the full pipeline:
//!
//! 1. Obtain a short-lived Xet read token from the Hub API.
//! 2. Resolve the Xet file hash for a given repository file.
//! 3. Fetch the reconstruction plan from the Xet CAS (Content-Addressable
//!    Storage) service.
//! 4. Download the required xorb byte-ranges.
//! 5. Deserialize and decompress individual chunks.
//! 6. Reassemble the chunks into the final output file.
//!
//! # Chunk binary format
//!
//! Each chunk inside a xorb byte-range is laid out as:
//!
//! ```text
//! byte 0      : version (u8, expected 0)
//! bytes 1-3   : compressed_size as 3-byte little-endian u32
//! byte 4      : compression_type (0=None, 1=LZ4, 2=BG4+LZ4)
//! bytes 5-7   : uncompressed_size as 3-byte little-endian u32
//! bytes 8..   : compressed_size bytes of (possibly compressed) payload
//! ```

use std::collections::HashMap;
use std::path::Path;

use anyhow::Context;
use indicatif::ProgressBar;
use serde::Deserialize;
use tokio::io::AsyncWriteExt;

// ---------------------------------------------------------------------------
// API response types
// ---------------------------------------------------------------------------

/// Short-lived Xet read token returned by the Hub.
///
/// The `accessToken` JSON field is renamed to match Rust's snake_case
/// convention.
#[derive(Debug, Deserialize)]
pub struct XetTokenResponse {
    /// Bearer token used for subsequent Xet CAS API calls.
    #[serde(rename = "accessToken")]
    pub access_token: String,

    /// Unix timestamp (seconds) at which the token expires.
    // Deserialized from the API response; callers may use it for token refresh
    // logic in the future. Suppressing dead_code since this is a public API field.
    #[allow(dead_code)]
    pub exp: u64,

    /// Base URL of the Xet CAS service (e.g. `https://cas.xethub.com`).
    #[serde(rename = "casUrl")]
    pub cas_url: String,
}

/// A half-open or closed byte/chunk range `[start, end)` or `[start, end]`
/// depending on context.
///
/// In chunk indices (`fi.range`, `term.range`) the range is **end-exclusive**.
/// In byte offsets (`fi.url_range`) the `end` value is passed verbatim to the
/// HTTP `Range` header, where it is **inclusive**.
#[derive(Debug, Deserialize, Clone)]
pub struct Range {
    pub start: u64,
    pub end: u64,
}

/// A single reconstruction term: one contiguous slice of the output file that
/// is assembled from a range of chunks identified by `hash`.
#[derive(Debug, Deserialize)]
pub struct ReconstructionTerm {
    /// Xorb hash that identifies which xorb blob contains these chunks.
    pub hash: String,

    /// Total uncompressed byte length contributed by this term to the output.
    // Part of the CAS API response; reserved for future validation logic.
    #[allow(dead_code)]
    pub unpacked_length: u64,

    /// Which chunk indices within the xorb blob to extract (end-exclusive).
    pub range: Range,
}

/// Location and byte range of a downloadable xorb segment.
#[derive(Debug, Deserialize)]
pub struct FetchInfo {
    /// Which chunk indices this xorb segment covers (end-exclusive).
    pub range: Range,

    /// Pre-signed URL from which the xorb segment can be downloaded.
    pub url: String,

    /// Byte range within the URL to request (used in the `Range:` header).
    /// `end` is **inclusive** (i.e. HTTP `bytes=start-end`).
    pub url_range: Range,
}

/// Full reconstruction plan for a single Xet file.
///
/// Returned by the CAS `/v1/reconstructions/{file_id}` endpoint.
#[derive(Debug, Deserialize)]
pub struct QueryReconstructionResponse {
    /// How many bytes to skip from the very first chunk of the first term.
    ///
    /// Non-zero when the file does not start on a chunk boundary.
    pub offset_into_first_range: u64,

    /// Ordered list of reconstruction terms.  Concatenating the decompressed
    /// chunks from each term (in order, honouring `offset_into_first_range`)
    /// yields the exact file content.
    pub terms: Vec<ReconstructionTerm>,

    /// Map from xorb hash to the list of downloadable segments that cover it.
    ///
    /// For a given `term`, look up `fetch_info[&term.hash]` and find the
    /// `FetchInfo` entry whose `range` contains `term.range`.
    pub fetch_info: HashMap<String, Vec<FetchInfo>>,
}

// ---------------------------------------------------------------------------
// Step 1 – Obtain a Xet read token
// ---------------------------------------------------------------------------

/// Fetch a short-lived Xet read token for a repository revision.
///
/// Calls `GET https://huggingface.co/api/{repo_type}s/{repo_id}/xet-read-token/{revision}`.
/// `repo_type` must be one of `"model"`, `"dataset"`, or `"space"` — it is
/// pluralised automatically (e.g. `"dataset"` → `/datasets/`).
/// An `Authorization: Bearer` header is added when `hub_token` is `Some`.
///
/// # Errors
///
/// - Network / TLS errors from [`reqwest`].
/// - Non-2xx HTTP status from the Hub API.
/// - JSON deserialisation failure.
pub async fn get_xet_token(
    client: &reqwest::Client,
    repo_id: &str,
    repo_type: &str,
    revision: &str,
    hub_token: Option<&str>,
) -> anyhow::Result<XetTokenResponse> {
    let url = format!(
        "https://huggingface.co/api/{repo_type}s/{repo_id}/xet-read-token/{revision}"
    );

    let mut request = client.get(&url);
    if let Some(token) = hub_token {
        request = request.bearer_auth(token);
    }

    let response = request
        .send()
        .await
        .with_context(|| format!("GET {url}: network error"))?;

    let status = response.status();
    if !status.is_success() {
        anyhow::bail!("GET {url} returned {status}");
    }

    response
        .json::<XetTokenResponse>()
        .await
        .with_context(|| format!("failed to deserialise Xet token response from {url}"))
}

// ---------------------------------------------------------------------------
// Step 2 – Resolve the Xet file hash
// ---------------------------------------------------------------------------

/// Probe the Hub redirect to extract the Xet file hash for `filename`.
///
/// Issues a `GET` to the resolve URL **without** following redirects and reads
/// the `X-Xet-Hash` response header.
///
/// Returns `Some(hash)` when the file is stored on Xet storage, `None`
/// otherwise (the caller should fall back to the standard LFS downloader).
///
/// # Errors
///
/// Returns an error only on network failure; a missing header yields `None`.
pub async fn get_xet_file_id(
    _client: &reqwest::Client,
    repo_id: &str,
    filename: &str,
    revision: &str,
    hub_token: Option<&str>,
) -> anyhow::Result<Option<String>> {
    // Build a separate one-shot client that never follows redirects.
    let no_redirect_client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .context("failed to build no-redirect HTTP client")?;

    let url = format!("https://huggingface.co/{repo_id}/resolve/{revision}/{filename}");

    let mut request = no_redirect_client.get(&url);
    if let Some(token) = hub_token {
        request = request.bearer_auth(token);
    }

    let response = request
        .send()
        .await
        .with_context(|| format!("GET {url}: network error"))?;

    // Extract the Xet hash header regardless of the HTTP status code (redirect
    // responses carry the header on 3xx, not 2xx).
    let hash = response
        .headers()
        .get("X-Xet-Hash")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_owned());

    Ok(hash)
}

// ---------------------------------------------------------------------------
// Step 3 – Fetch the reconstruction plan
// ---------------------------------------------------------------------------

/// Retrieve the chunk reconstruction plan for a Xet file.
///
/// Calls `GET {cas_url}/v1/reconstructions/{file_id}` with the Xet bearer
/// token.
///
/// # Errors
///
/// - Network / TLS errors.
/// - Non-2xx HTTP status from the CAS service.
/// - JSON deserialisation failure.
pub async fn get_reconstruction(
    client: &reqwest::Client,
    cas_url: &str,
    xet_token: &str,
    file_id: &str,
) -> anyhow::Result<QueryReconstructionResponse> {
    let url = format!("{cas_url}/v1/reconstructions/{file_id}");

    let response = client
        .get(&url)
        .bearer_auth(xet_token)
        .send()
        .await
        .with_context(|| format!("GET {url}: network error"))?;

    let status = response.status();
    if !status.is_success() {
        anyhow::bail!("GET {url} returned {status}");
    }

    response
        .json::<QueryReconstructionResponse>()
        .await
        .with_context(|| format!("failed to deserialise reconstruction response from {url}"))
}

// ---------------------------------------------------------------------------
// Step 4 – Download a xorb byte range
// ---------------------------------------------------------------------------

/// Download a byte range from a pre-signed xorb URL.
///
/// Uses an HTTP `Range: bytes={start}-{end}` header where `end` is
/// **inclusive**, matching the semantics of `FetchInfo::url_range`.
///
/// # Errors
///
/// - Network / TLS errors.
/// - Non-2xx / non-206 HTTP status.
async fn download_xorb_range(
    client: &reqwest::Client,
    url: &str,
    start: u64,
    end: u64,
) -> anyhow::Result<Vec<u8>> {
    let range_header = format!("bytes={start}-{end}");

    let response = client
        .get(url)
        .header(reqwest::header::RANGE, &range_header)
        .send()
        .await
        .with_context(|| format!("GET {url} ({range_header}): network error"))?;

    let status = response.status();
    // 200 OK is also acceptable when the server returns the full body instead
    // of honouring the range request, but 206 Partial Content is the norm.
    if !status.is_success() {
        anyhow::bail!("GET {url} ({range_header}) returned {status}");
    }

    response
        .bytes()
        .await
        .map(|b| b.to_vec())
        .with_context(|| format!("failed to read body from {url}"))
}

// ---------------------------------------------------------------------------
// Step 5 – Deserialize xorb chunks
// ---------------------------------------------------------------------------

/// Reverse the BG4 (byte-grouping) permutation applied before LZ4 compression.
///
/// BG4 groups bytes at positions `0,4,8,…` then `1,5,9,…` then `2,6,10,…`
/// then `3,7,11,…` into contiguous runs to improve LZ4 compression ratios.
/// This function inverts that permutation.
fn unbg4(data: &[u8]) -> Vec<u8> {
    let n = data.len();
    let full_groups = n / 4;
    let remainder = n % 4;
    let mut out = vec![0u8; n];

    // `group_start` accumulates the byte offset into `data` where each of the
    // four interleaved streams begins.
    let mut group_start: usize = 0;

    for i in 0..4usize {
        // Stream `i` carries bytes at output positions `i, i+4, i+8, …`.
        // When `n` is not divisible by 4, the first `remainder` streams each
        // have one extra element.
        let group_size = full_groups + if i < remainder { 1 } else { 0 };

        for j in 0..group_size {
            out[j * 4 + i] = data[group_start + j];
        }

        group_start += group_size;
    }

    out
}

/// Deserialize and decompress chunks from a raw xorb binary payload.
///
/// Parses chunk headers sequentially from byte offset 0.  Only chunks whose
/// index falls in `[chunk_range.start, chunk_range.end)` are decompressed and
/// returned; earlier chunks are skipped by advancing the read cursor without
/// decompressing them.
///
/// # Errors
///
/// - Truncated header or body (fewer bytes remaining than the header declares).
/// - Unknown compression type.
/// - LZ4 decompression failure.
/// - Output length mismatch after decompression.
fn deserialize_xorb_chunks(data: &[u8], chunk_range: &Range) -> anyhow::Result<Vec<Vec<u8>>> {
    const HEADER_LEN: usize = 8;

    let mut cursor: usize = 0;
    let mut chunk_index: u64 = 0;
    let mut result: Vec<Vec<u8>> = Vec::new();

    while chunk_index < chunk_range.end {
        // Ensure we have at least a full 8-byte header.
        if cursor + HEADER_LEN > data.len() {
            anyhow::bail!(
                "xorb truncated at chunk {chunk_index}: \
                 need {HEADER_LEN} header bytes but only {} remain",
                data.len().saturating_sub(cursor)
            );
        }

        let header = &data[cursor..cursor + HEADER_LEN];
        cursor += HEADER_LEN;

        // byte 0: format version (must be 0).
        let version = header[0];
        if version != 0 {
            anyhow::bail!(
                "unknown xorb chunk version {version} at chunk {chunk_index}"
            );
        }

        // bytes 1-3: compressed_size as 3-byte little-endian u32.
        let compressed_size = u32::from_le_bytes([header[1], header[2], header[3], 0]) as usize;

        // byte 4: compression_type.
        let compression_type = header[4];

        // bytes 5-7: uncompressed_size as 3-byte little-endian u32.
        let uncompressed_size = u32::from_le_bytes([header[5], header[6], header[7], 0]) as usize;

        // Ensure the declared payload is present in `data`.
        if cursor + compressed_size > data.len() {
            anyhow::bail!(
                "xorb truncated at chunk {chunk_index}: \
                 need {compressed_size} payload bytes but only {} remain",
                data.len().saturating_sub(cursor)
            );
        }

        let payload = &data[cursor..cursor + compressed_size];
        cursor += compressed_size;

        // Only decompress and keep chunks that fall inside the requested range.
        if chunk_index >= chunk_range.start {
            let decompressed = match compression_type {
                // No compression — payload is the raw output.
                0 => {
                    if compressed_size != uncompressed_size {
                        anyhow::bail!(
                            "chunk {chunk_index}: compression_type=None but \
                             compressed_size({compressed_size}) != \
                             uncompressed_size({uncompressed_size})"
                        );
                    }
                    payload.to_vec()
                }

                // LZ4 — decompress directly.
                1 => lz4_flex::decompress(payload, uncompressed_size).with_context(|| {
                    format!("LZ4 decompression failed for chunk {chunk_index}")
                })?,

                // BG4 + LZ4 — LZ4-decompress then reverse the byte-grouping permutation.
                2 => {
                    let lz4_out =
                        lz4_flex::decompress(payload, uncompressed_size).with_context(|| {
                            format!(
                                "LZ4 decompression (BG4 stage) failed for chunk {chunk_index}"
                            )
                        })?;
                    unbg4(&lz4_out)
                }

                other => {
                    anyhow::bail!(
                        "unknown compression_type {other} in chunk {chunk_index}"
                    );
                }
            };

            if decompressed.len() != uncompressed_size {
                anyhow::bail!(
                    "chunk {chunk_index}: decompressed length {} != \
                     declared uncompressed_size {uncompressed_size}",
                    decompressed.len()
                );
            }

            result.push(decompressed);
        }
        // Chunks before chunk_range.start are skipped (payload already advanced).

        chunk_index += 1;
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Step 6 – Main download entry point
// ---------------------------------------------------------------------------

/// Download a single file via the Xet protocol and write it to `dest`.
///
/// This is the high-level orchestration function that chains all protocol
/// steps together:
///
/// 1. Fetches a short-lived Xet read token.
/// 2. Resolves the Xet file hash; returns an error if the file is not on Xet.
/// 3. Retrieves the chunk reconstruction plan from the CAS service.
/// 4. Creates the output file (parent directories are created as needed).
/// 5. For each reconstruction term: locates the covering `FetchInfo`, downloads
///    the xorb byte range, deserialises the chunks, extracts those belonging to
///    the term, skips `offset_into_first_range` bytes for the very first chunk,
///    writes the bytes to the output file, and advances the progress bar.
/// 6. Flushes and closes the output file; finishes the progress bar.
///
/// # Errors
///
/// - Hub or CAS API errors (network, auth, HTTP status).
/// - File I/O errors.
/// - Xorb binary parsing or decompression errors.
pub async fn download_file_xet(
    hub_client: &reqwest::Client,
    repo_id: &str,
    repo_type: &str,
    filename: &str,
    revision: &str,
    dest: &Path,
    hub_token: Option<&str>,
    progress_bar: &ProgressBar,
) -> anyhow::Result<()> {
    // ------------------------------------------------------------------
    // Step 1: Obtain a Xet read token.
    // ------------------------------------------------------------------
    let xet_token_response = get_xet_token(hub_client, repo_id, repo_type, revision, hub_token)
        .await
        .with_context(|| format!("failed to get Xet token for {repo_id}@{revision}"))?;

    // ------------------------------------------------------------------
    // Step 2: Resolve the Xet file hash.
    // ------------------------------------------------------------------
    let file_id = get_xet_file_id(hub_client, repo_id, filename, revision, hub_token)
        .await
        .with_context(|| format!("failed to resolve Xet file ID for {filename}"))?
        .ok_or_else(|| anyhow::anyhow!("File not on Xet storage: {filename}"))?;

    // ------------------------------------------------------------------
    // Step 3: Fetch the reconstruction plan.
    // ------------------------------------------------------------------
    let reconstruction = get_reconstruction(
        hub_client,
        &xet_token_response.cas_url,
        &xet_token_response.access_token,
        &file_id,
    )
    .await
    .with_context(|| format!("failed to get reconstruction for file_id={file_id}"))?;

    // ------------------------------------------------------------------
    // Step 4: Create the output file.
    // ------------------------------------------------------------------
    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("failed to create directories for {}", dest.display()))?;
    }

    let mut output_file = tokio::fs::File::create(dest)
        .await
        .with_context(|| format!("failed to create output file {}", dest.display()))?;

    // ------------------------------------------------------------------
    // Step 5: Process each reconstruction term.
    // ------------------------------------------------------------------
    let mut is_first_term = true;

    for term in &reconstruction.terms {
        // Find the FetchInfo entry whose chunk range fully contains term.range.
        let fetch_infos = reconstruction
            .fetch_info
            .get(&term.hash)
            .with_context(|| {
                format!(
                    "no fetch_info entry for hash {} (file_id={file_id})",
                    term.hash
                )
            })?;

        let fi = fetch_infos
            .iter()
            .find(|fi| fi.range.start <= term.range.start && fi.range.end >= term.range.end)
            .with_context(|| {
                format!(
                    "no covering FetchInfo for term hash={} range=[{},{})",
                    term.hash, term.range.start, term.range.end
                )
            })?;

        // Download the raw xorb bytes for this segment.
        let raw_data =
            download_xorb_range(hub_client, &fi.url, fi.url_range.start, fi.url_range.end)
                .await
                .with_context(|| {
                    format!(
                        "failed to download xorb for hash={} url_range=[{},{}]",
                        term.hash, fi.url_range.start, fi.url_range.end
                    )
                })?;

        // Deserialise all chunks covered by fi.range, then keep only those in
        // [term.range.start, term.range.end).
        //
        // fi.range is the full range available in the downloaded payload.
        // term.range is a subset of fi.range that this term needs.
        let all_chunks = deserialize_xorb_chunks(&raw_data, &fi.range)
            .with_context(|| format!("failed to deserialise xorb chunks for hash={}", term.hash))?;

        // Compute the slice of `all_chunks` that corresponds to term.range.
        // `all_chunks[0]` corresponds to chunk index `fi.range.start`.
        let term_start_in_all = (term.range.start - fi.range.start) as usize;
        let term_end_in_all = (term.range.end - fi.range.start) as usize;

        let term_chunks = all_chunks
            .get(term_start_in_all..term_end_in_all)
            .with_context(|| {
                format!(
                    "term chunk slice [{term_start_in_all}..{term_end_in_all}) \
                     out of bounds for all_chunks len={} (hash={})",
                    all_chunks.len(),
                    term.hash
                )
            })?;

        let mut bytes_written: u64 = 0;

        for (chunk_idx, chunk) in term_chunks.iter().enumerate() {
            let write_slice: &[u8] = if is_first_term && chunk_idx == 0 {
                // For the very first chunk of the very first term, skip
                // `offset_into_first_range` bytes as directed by the
                // reconstruction plan.
                let offset = reconstruction.offset_into_first_range as usize;
                if offset > chunk.len() {
                    anyhow::bail!(
                        "offset_into_first_range ({offset}) exceeds first chunk \
                         length ({})",
                        chunk.len()
                    );
                }
                &chunk[offset..]
            } else {
                chunk.as_slice()
            };

            output_file
                .write_all(write_slice)
                .await
                .with_context(|| {
                    format!("failed to write chunk to {}", dest.display())
                })?;

            bytes_written += write_slice.len() as u64;
        }

        progress_bar.inc(bytes_written);
        is_first_term = false;
    }

    // Flush all buffered writes to disk.
    output_file
        .flush()
        .await
        .with_context(|| format!("failed to flush output file {}", dest.display()))?;

    // ------------------------------------------------------------------
    // Step 6: Finish the progress bar.
    // ------------------------------------------------------------------
    progress_bar.finish_with_message(format!("downloaded {filename}"));

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // unbg4 tests
    // -----------------------------------------------------------------------

    /// Four bytes — the trivial case: one element per stream, no remainder.
    #[test]
    fn unbg4_four_bytes() {
        // BG4 over [0,1,2,3] groups all bytes at positions 0,4,…; 1,5,…;
        // 2,6,…; 3,7,… For 4 bytes there is exactly one element per stream.
        // Streams: [byte@0=0], [byte@1=1], [byte@2=2], [byte@3=3].
        // Encoded: [0, 1, 2, 3].
        // Decoded: [0, 1, 2, 3] (identity for 4 bytes).
        let input = [0u8, 1, 2, 3];
        let output = unbg4(&input);
        assert_eq!(output, vec![0, 1, 2, 3]);
    }

    /// Eight bytes — two full groups of four, no remainder.
    #[test]
    fn unbg4_eight_bytes() {
        // Original: [A0, A1, A2, A3, B0, B1, B2, B3]
        // BG4 encodes as: [A0, B0, A1, B1, A2, B2, A3, B3]
        // unbg4([A0, B0, A1, B1, A2, B2, A3, B3]) => [A0, A1, A2, A3, B0, B1, B2, B3]
        let encoded = [b'A', b'E', b'B', b'F', b'C', b'G', b'D', b'H'];
        // Streams of length 2 each:
        //   stream 0: A, E  -> positions 0, 4
        //   stream 1: B, F  -> positions 1, 5
        //   stream 2: C, G  -> positions 2, 6
        //   stream 3: D, H  -> positions 3, 7
        let expected = [b'A', b'B', b'C', b'D', b'E', b'F', b'G', b'H'];
        assert_eq!(unbg4(&encoded), expected);
    }

    /// Five bytes — one remainder byte goes to stream 0.
    #[test]
    fn unbg4_five_bytes_remainder() {
        // n=5, full_groups=1, remainder=1
        // stream 0 size=2, stream 1 size=1, stream 2 size=1, stream 3 size=1
        // encoded layout: [s0[0], s0[1], s1[0], s2[0], s3[0]]
        //                  pos0    pos4   pos1    pos2    pos3
        let encoded = [10u8, 50, 20, 30, 40];
        // stream 0 (positions 0,4): 10, 50  -> out[0]=10, out[4]=50
        // stream 1 (position 1):    20      -> out[1]=20
        // stream 2 (position 2):    30      -> out[2]=30
        // stream 3 (position 3):    40      -> out[3]=40
        let expected = [10u8, 20, 30, 40, 50];
        assert_eq!(unbg4(&encoded), expected);
    }

    /// Empty input should produce empty output without panicking.
    #[test]
    fn unbg4_empty() {
        assert_eq!(unbg4(&[]), Vec::<u8>::new());
    }

    // -----------------------------------------------------------------------
    // deserialize_xorb_chunks tests
    // -----------------------------------------------------------------------

    /// Helper to build a minimal xorb chunk header + payload.
    fn make_chunk(payload: &[u8], compression_type: u8) -> Vec<u8> {
        let compressed_size = payload.len();
        let uncompressed_size = payload.len(); // only valid for compression_type=0

        let mut chunk = vec![0u8; 8 + compressed_size];
        chunk[0] = 0; // version
        // bytes 1-3: compressed_size little-endian 3-byte
        let cs_bytes = compressed_size.to_le_bytes();
        chunk[1] = cs_bytes[0];
        chunk[2] = cs_bytes[1];
        chunk[3] = cs_bytes[2];
        chunk[4] = compression_type;
        // bytes 5-7: uncompressed_size little-endian 3-byte
        let us_bytes = uncompressed_size.to_le_bytes();
        chunk[5] = us_bytes[0];
        chunk[6] = us_bytes[1];
        chunk[7] = us_bytes[2];
        chunk[8..].copy_from_slice(payload);
        chunk
    }

    #[test]
    fn deserialize_single_uncompressed_chunk() {
        let payload = b"hello";
        let data = make_chunk(payload, 0);

        let range = Range { start: 0, end: 1 };
        let chunks = deserialize_xorb_chunks(&data, &range).expect("should parse");

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], b"hello");
    }

    #[test]
    fn deserialize_skips_chunks_before_range_start() {
        // Two chunks; request only the second one [1..2).
        let chunk0 = make_chunk(b"first", 0);
        let chunk1 = make_chunk(b"second", 0);
        let mut data = chunk0;
        data.extend_from_slice(&chunk1);

        let range = Range { start: 1, end: 2 };
        let chunks = deserialize_xorb_chunks(&data, &range).expect("should parse");

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], b"second");
    }

    #[test]
    fn deserialize_multiple_chunks() {
        let chunk0 = make_chunk(b"alpha", 0);
        let chunk1 = make_chunk(b"beta", 0);
        let chunk2 = make_chunk(b"gamma", 0);

        let mut data = chunk0;
        data.extend_from_slice(&chunk1);
        data.extend_from_slice(&chunk2);

        let range = Range { start: 0, end: 3 };
        let chunks = deserialize_xorb_chunks(&data, &range).expect("should parse");

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], b"alpha");
        assert_eq!(chunks[1], b"beta");
        assert_eq!(chunks[2], b"gamma");
    }

    #[test]
    fn deserialize_truncated_header_returns_error() {
        // Only 4 bytes — not enough for an 8-byte header.
        let data = [0u8, 4, 0, 0];
        let range = Range { start: 0, end: 1 };
        assert!(deserialize_xorb_chunks(&data, &range).is_err());
    }

    #[test]
    fn deserialize_truncated_payload_returns_error() {
        let mut data = make_chunk(b"hello", 0);
        // Truncate by removing the last byte so the payload is incomplete.
        data.pop();

        let range = Range { start: 0, end: 1 };
        assert!(deserialize_xorb_chunks(&data, &range).is_err());
    }

    #[test]
    fn deserialize_unknown_version_returns_error() {
        let mut data = make_chunk(b"test", 0);
        data[0] = 99; // corrupt version byte

        let range = Range { start: 0, end: 1 };
        assert!(deserialize_xorb_chunks(&data, &range).is_err());
    }

    #[test]
    fn deserialize_unknown_compression_type_returns_error() {
        let mut data = make_chunk(b"test", 0);
        data[4] = 42; // unsupported compression type

        let range = Range { start: 0, end: 1 };
        assert!(deserialize_xorb_chunks(&data, &range).is_err());
    }

    #[test]
    fn deserialize_lz4_chunk() {
        let payload = b"the quick brown fox jumps over the lazy dog";
        let compressed = lz4_flex::compress_prepend_size(payload);

        // Build the header manually: compression_type=1 (LZ4).
        // The stored payload for decompression must NOT include the prepended
        // size prefix, so we use the raw compress variant.
        let raw_compressed = lz4_flex::compress(payload);
        let mut data = vec![0u8; 8 + raw_compressed.len()];
        data[0] = 0; // version
        let cs_bytes = raw_compressed.len().to_le_bytes();
        data[1] = cs_bytes[0];
        data[2] = cs_bytes[1];
        data[3] = cs_bytes[2];
        data[4] = 1; // LZ4
        let us_bytes = payload.len().to_le_bytes();
        data[5] = us_bytes[0];
        data[6] = us_bytes[1];
        data[7] = us_bytes[2];
        data[8..].copy_from_slice(&raw_compressed);

        // Suppress unused variable warning — compress_prepend_size is called
        // for illustrative purposes above.
        let _ = compressed;

        let range = Range { start: 0, end: 1 };
        let chunks = deserialize_xorb_chunks(&data, &range).expect("LZ4 decompression failed");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], payload);
    }

    // -----------------------------------------------------------------------
    // Network integration test
    // -----------------------------------------------------------------------

    /// End-to-end integration test: download the Xet protocol reference CSV file
    /// from the public `xet-team/xet-spec-reference-files` dataset, verify its
    /// SHA-256 matches the known value from the HuggingFace Hub API, and clean up.
    ///
    /// Run with:
    ///   cargo test test_xet_download_reference_csv -- --ignored --nocapture
    #[tokio::test]
    #[ignore = "requires network; run with: cargo test -- --ignored"]
    async fn test_xet_download_reference_csv() {
        use sha2::{Digest, Sha256};
        use std::io::Read;

        const REPO_ID: &str = "xet-team/xet-spec-reference-files";
        const REPO_TYPE: &str = "dataset";
        const FILENAME: &str = "Electric_Vehicle_Population_Data_20250917.csv";
        const REVISION: &str = "main";
        const EXPECTED_SIZE: u64 = 63_527_244;
        // SHA-256 from:
        //   GET https://huggingface.co/api/datasets/xet-team/xet-spec-reference-files?blobs=true
        //   siblings[*].lfs.sha256 for the CSV entry
        const EXPECTED_SHA256: &str =
            "f41255b252f776125f2b657136654e4d4d5d2ccf8ef4db0ec186bd1981b69734";

        let client = reqwest::Client::builder()
            .user_agent("HFExport/0.1 integration-test")
            .build()
            .expect("failed to build reqwest client");

        let dest = std::env::temp_dir()
            .join("hfexport_integration_test")
            .join(FILENAME);

        // Remove any leftover file from a previous interrupted run.
        if dest.exists() {
            std::fs::remove_file(&dest).expect("failed to remove stale test file");
        }

        let pb = indicatif::ProgressBar::new(EXPECTED_SIZE);
        pb.set_style(
            indicatif::ProgressStyle::with_template(
                "{spinner} {wide_msg} [{bar:40}] {bytes}/{total_bytes} ({bytes_per_sec})",
            )
            .unwrap(),
        );
        pb.set_message(FILENAME);

        download_file_xet(
            &client,
            REPO_ID,
            REPO_TYPE,
            FILENAME,
            REVISION,
            &dest,
            None, // public repo — no auth token needed
            &pb,
        )
        .await
        .expect("download_file_xet failed");

        pb.finish_with_message("done");

        // -- Verify file size -------------------------------------------------------
        let metadata = std::fs::metadata(&dest).expect("failed to stat output file");
        assert_eq!(
            metadata.len(),
            EXPECTED_SIZE,
            "downloaded file has wrong size: expected {EXPECTED_SIZE}, got {}",
            metadata.len()
        );

        // -- Verify SHA-256 ---------------------------------------------------------
        let mut file = std::fs::File::open(&dest).expect("failed to open output file");
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 65_536];
        loop {
            let n = file.read(&mut buf).expect("read error");
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        let actual_sha256 = hex::encode(hasher.finalize());
        assert_eq!(
            actual_sha256, EXPECTED_SHA256,
            "SHA-256 mismatch: expected {EXPECTED_SHA256}, got {actual_sha256}"
        );

        // -- Clean up ---------------------------------------------------------------
        std::fs::remove_file(&dest).expect("failed to remove test file");
    }
}
