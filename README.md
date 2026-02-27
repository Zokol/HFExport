# HFExport

A fast, resumable CLI tool for downloading model repositories from the [HuggingFace Hub](https://huggingface.co), written in Rust.

Downloads use the [Xet storage protocol](https://huggingface.co/docs/hub/xet/index) automatically when available — a chunk-based content-addressed system that is faster than standard LFS transfers. Files not on Xet storage fall back to standard HTTP range downloads transparently.

## Features

- **Automatic Xet downloads** — uses HuggingFace's fast chunk-based Xet protocol when the file supports it; falls back to LFS silently
- **Resumable transfers** — interrupted downloads continue from where they stopped
- **Batch downloads** — download multiple repositories in one command
- **Per-file progress bars** — live speed, ETA, and byte counters via `indicatif`
- **Authenticated downloads** — token stored securely at `~/.cache/hfexport/token` with `0o600` permissions, never in the project directory
- **Structured output** — files saved as `{output}/{owner}/{model-name}/{filename}`

## Installation

Requires [Rust](https://rustup.rs) (edition 2024, Rust 1.85+).

```bash
git clone https://github.com/yourname/HFExport
cd HFExport
cargo build --release
```

The binary is at `target/release/HFExport`. Optionally copy it to a directory on your `$PATH`:

```bash
cp target/release/HFExport ~/.local/bin/hfexport
```

## Usage

### Download public models (no token required)

```bash
hfexport download -m microsoft/phi-2
```

### Download multiple models at once

```bash
hfexport download -m microsoft/phi-2 -m mistralai/Mistral-7B-v0.1
```

### Specify an output directory

```bash
hfexport download -m meta-llama/Llama-3.2-1B -o /data/models
```

Default output directory is `./models` relative to the current working directory.

### Authenticated downloads (private or gated models)

Save your [HuggingFace API token](https://huggingface.co/settings/tokens) first:

```bash
hfexport login
# Paste your HuggingFace token (input hidden):
```

The token is read from `/dev/tty` so it is never echoed, even when output is piped. It is saved to `~/.cache/hfexport/token` and loaded automatically on subsequent runs.

## Output structure

```
./models/
└── microsoft/
    └── phi-2/
        ├── config.json
        ├── model.safetensors
        ├── tokenizer.json
        └── ...
```

Each repository gets its own `{owner}/{model-name}` subdirectory. Files that already exist at the expected size are skipped without re-downloading.

## How downloads work

For each file in a repository:

1. **Xet fast-path** — queries the HuggingFace Hub for a Xet file hash (`X-Xet-Hash`). If the file is on Xet storage, fetches the CAS reconstruction plan and downloads the required xorb chunk ranges in parallel, decompresses them (None / LZ4 / BG4+LZ4), and assembles the output file.
2. **LFS fallback** — if the file is not on Xet storage (or Xet is unavailable), falls back to a standard HTTP range download with resume support.

No configuration is needed to use Xet — it is attempted automatically.

## Command reference

```
hfexport <COMMAND>

Commands:
  login     Save a HuggingFace API token for authenticated downloads
  download  Download one or more HuggingFace model repositories

Options for download:
  -m, --models <REPO>...   One or more repo IDs (e.g. microsoft/phi-2)  [required]
  -o, --output <DIR>       Output directory  [default: ./models]
  -h, --help               Print help
```

## Running tests

```bash
# Unit tests (offline, fast)
cargo test

# End-to-end integration test (requires network, downloads ~63 MB)
cargo test test_xet_download_reference_csv -- --ignored --nocapture
```

The integration test downloads the [Xet protocol reference CSV](https://huggingface.co/datasets/xet-team/xet-spec-reference-files) and verifies its SHA-256 against the known value from the HuggingFace Hub API.
