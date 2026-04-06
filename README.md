

# QORA-STT - Pure Rust Speech-to-Text

Pure Rust inference engine for OpenAI's Whisper Tiny. No Python, no CUDA, no external dependencies. Single executable + binary weights = portable speech-to-text on any machine.

Based on **openai/whisper-tiny** (MIT License).

## Quick Start

```bash
# Transcribe an audio file (English)
qora-stt.exe --model-path . --load model.qora-stt --audio recording.wav

# Specify language
qora-stt.exe --model-path . --load model.qora-stt --audio recording.wav --language french

# Save transcription to file
qora-stt.exe --model-path . --load model.qora-stt --audio recording.wav --output transcript.txt
```

## Files

```
model/
  qora-stt.exe       2.5 MB    Inference engine (single binary)
  model.qora-stt     144 MB    F32 weights (encoder + decoder)
  config.json         2.0 KB   Model configuration
  tokenizer.json      2.4 MB   Tokenizer (51,865 vocab)
  README.md                    This file
```

**No safetensors needed.** Everything loads from `model.qora-stt`.

## Model Info

| Property | Value |
|----------|-------|
| **Base Model** | openai/whisper-tiny |
| **Parameters** | 39 Million |
| **Type** | Encoder-decoder transformer |
| **Weights** | F32 (no quantization needed at 39M params) |
| **Binary Size** | 144 MB |
| **Input** | WAV audio (any sample rate, auto-resampled to 16kHz) |
| **Output** | Transcribed text |
| **Max Duration** | 30 seconds per chunk |
| **Languages** | 99 languages supported |
| **Platform** | Windows x86_64, Linux x86_64, macOS aarch64 |

## Architecture

| Component | Details |
|-----------|---------|
| **Encoder** | Conv1D stem (80->384, stride 2) + 4 transformer layers |
| **Decoder** | 4 transformer layers with cross-attention to encoder |
| **Hidden Size** | 384 |
| **Attention Heads** | 6 (head_dim=64) |
| **FFN Dimension** | 1,536 |
| **Vocabulary** | 51,865 tokens (BPE) |
| **Activation** | GELU |
| **Normalization** | LayerNorm with bias |
| **Mel Spectrogram** | 80 bins, n_fft=400, hop=160, 16kHz |
| **Position Encoding** | Encoder: sinusoidal (stored), Decoder: learned |

### Encoder
1. **Conv1D stem**: Conv1(80->384, k=3, s=1) -> GELU -> Conv2(384->384, k=3, s=2) -> GELU
2. Input: mel spectrogram `[80, 3000]` -> output `[1500, 384]`
3. 4 transformer layers: LayerNorm -> self-attention (6 heads, full) -> residual -> LayerNorm -> FFN -> residual
4. Final LayerNorm

### Decoder (Autoregressive)
1. Token + positional embedding
2. 4 transformer layers, each with:
   - Causal self-attention (with KV cache)
   - Cross-attention to encoder output (cached once)
   - FFN (384 -> 1536 -> 384)
3. Output projection (tied with token embeddings)

## CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path <dir>` | `.` | Directory with config.json + tokenizer.json |
| `--load <path>` | -- | Load binary model (.qora-stt) |
| `--audio <wav>` | -- | Input WAV file to transcribe |
| `--language <name>` | english | Language name or code (e.g., "french", "fr") |
| `--output <path>` | -- | Write transcription to text file |
| `--save <path>` | -- | Save binary model (for converting from safetensors) |
| `--help` | -- | Show help |

## Supported Languages

99 languages including: English, Chinese, German, Spanish, Russian, Korean, French, Japanese, Portuguese, Turkish, Polish, Dutch, Arabic, Swedish, Italian, Indonesian, Hindi, Finnish, Vietnamese, Hebrew, Ukrainian, Greek, Czech, Romanian, Danish, Hungarian, Tamil, Norwegian, Thai, Urdu, Croatian, Bulgarian, Lithuanian, Latin, Malayalam, Welsh, Slovak, Telugu, Persian, Latvian, Bengali, Serbian, Azerbaijani, Slovenian, Kannada, Estonian, Macedonian, Breton, Basque, Icelandic, Armenian, Nepali, Mongolian, Bosnian, Kazakh, Albanian, Swahili, Galician, Marathi, Punjabi, Sinhala, Khmer, Shona, Yoruba, Somali, Afrikaans, Occitan, Georgian, Belarusian, Tajik, Sindhi, Gujarati, Amharic, Yiddish, Lao, Uzbek, Faroese, Haitian, Pashto, Turkmen, Nynorsk, Maltese, Sanskrit, Luxembourgish, Myanmar, Tibetan, Tagalog, Malagasy, Assamese, Tatar, Hawaiian, Lingala, Hausa, Bashkir, Javanese, Sundanese.

## Performance (i5-11500, 16GB RAM, CPU-only)

| Phase | Time |
|-------|------|
| Model Load (binary) | ~92ms |
| Mel Extraction | ~108ms |
| Encoder (4 layers) | ~2.6s |
| Cross-attention Cache | ~32ms |
| Decoding | ~26ms/token |
| **Total (6s audio, 21 tokens)** | **~3.5s** |
| Memory | ~144 MB |

### Optimizations

- **Rayon parallelism**: GEMM rows parallelized across all CPU cores
- **Cache-friendly GEMM**: i-p-j loop order for sequential memory access
- **Parallel attention heads**: 6 heads computed concurrently
- **KV caching**: Cross-attention K/V computed once, reused every decoder step
- **Self-attention cache**: Grows incrementally, no recomputation

### AVX-512 SIMD Acceleration

On CPUs with AVX-512 support (Intel 11th gen+, AMD Zen 4+), QORA-STT automatically uses hand-written AVX-512 SIMD kernels:

| Kernel | Technique | Speedup |
|--------|-----------|---------|
| **F32 GEMV** | `fmadd_ps` FMA with 16 f32 values per cycle | ~2x |
| **F32 GEMV+bias** | Fused multiply-add with bias addition | ~2x |
| **F32 GEMM row** | Vectorized single-row accumulation | ~2x |

Detection is automatic at runtime — falls back to scalar code on non-AVX-512 CPUs with zero overhead.

## Converting from Safetensors

If you have the original `openai/whisper-tiny` safetensors:

```bash
# Download model
huggingface-cli download openai/whisper-tiny --local-dir whisper-tiny

# Convert to binary (runs one dummy transcription to trigger save)
qora-stt.exe --model-path whisper-tiny --save model.qora-stt --audio some.wav
```

After conversion, safetensors files are no longer needed.

## Platform Support

| Platform | Binary | Status |
|----------|--------|--------|
| **Windows x86_64** | `qora-stt.exe` | Tested |
| **Linux x86_64** | `qora-stt` | Supported |
| **macOS aarch64** | `qora-stt` | Supported |

CPU-only — no GPU needed. Pre-built binaries on the [Releases](https://github.com/qora-protocol/QORA-STT-TINY/releases) page.

## Building from Source

```bash
cargo build --release
```

### Dependencies

- **Language**: Pure Rust (2021 edition)
- `half` — F16 support
- `tokenizers` — HuggingFace tokenizer
- `safetensors` — Weight loading (for conversion only)
- `serde_json` — Config parsing
- `rayon` — Parallel computation
- **No ML framework** — all matrix ops are hand-written Rust

## QORA Model Family

| Engine | Model | Params | Size | Purpose |
|--------|-------|--------|------|---------|
| **QORA** | SmolLM3-3B | 3.07B | 1.68 GB (Q4) | Text generation, reasoning, chat |
| **QORA-TTS** | Qwen3-TTS-12Hz | 0.6B/1.7B | 1.5 GB (Q4) | Text-to-speech synthesis |
| **QORA-STT** | Whisper Tiny | 39M | 144 MB (F32) | Speech-to-text transcription |
| **QORA-Image** | SDXS-512 | 350M | 350 MB | Text-to-image generation |
| **QORA-Vision (Image)** | SigLIP 2 Base | 93M | 210 MB (Q4) | Image embeddings, zero-shot classification |
| **QORA-Vision (Video)** | ViViT Base | 89M | 60 MB (Q4) | Video action classification |

All engines are pure Rust, single-binary executables with no Python dependencies.

## License

The QORA-STT inference engine is custom-built. The Whisper Tiny model weights are released under the [MIT License](https://github.com/openai/whisper/blob/main/LICENSE) by OpenAI.

---

*Built with QORA - Pure Rust AI Inference*
