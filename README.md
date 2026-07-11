# Anime Super-Resolution Video Player

Real-time GPU-resident anime video upscaler. Decodes a video file, upscales every frame through a trained super-resolution model running on TensorRT, and displays the result — all on-GPU with zero CPU-GPU memory copies in the processing path.

Built on DeepStream 9.0 / GStreamer for the media pipeline, with a custom `nvdsvideotemplate` plugin that runs a TensorRT engine for the actual SR inference, and a GTK file-picker GUI for playback.

Tested on Ubuntu 24.04 with an RTX 3080 (12GB). Runs at 65-80 FPS on 720×480 anime content with the 2x model — well above the 30fps needed for real-time playback.

## Demo



https://github.com/user-attachments/assets/9c983ae0-6e14-4ec8-93cc-3d9f13eef664



## Project structure

```
.
├── model/
│   ├── apisr.py              # Model architectures (4 variants)
│   ├── train.py              # Training loop, ONNX export, evaluation
│   ├── dataset.py            # Patch-based SR dataset with augmentation
│   ├── quant.py              # TRT engine builder + ORT/TRT benchmark suite
│   └── trt_engines/          # Built .trt engine files (not checked in)
│
├── plugin/
│   ├── customlib_impl.cpp    # nvdsvideotemplate custom library (the DeepStream plugin)
│   ├── sr_trt_engine.hpp/cpp # TRT engine loader with runtime scale detection
│   ├── sr_pipeline.hpp/cpp   # Per-frame tiling + cosine-blend pipeline
│   ├── tile_kernels.cuh/cu   # Batch tile extraction + stitch CUDA kernels
│   ├── pack_rgba_kernel.cuh/cu  # RGB planes → RGBA interleave kernel
│   └── Makefile              # Builds libcustom_videoimpl.so
│
├── player/
│   └── player.py             # GTK GUI: file picker, seek bar, audio, upscaled playback
│
├── .gitignore
└── README.md
```


## How it works

### The model

APISR is a single-image super-resolution network trained specifically on anime content. It takes a low-resolution RGB patch (e.g. 64×64) and outputs a high-resolution version (128×128 at 2x, or 256×256 at 4x). The architecture is a standard residual network with a PixelShuffle upsampler, available in four variants trading quality against speed:

| Variant | Tag | Params | Block type | Expected speed (720p, TRT FP16) |
|---------|-----|--------|------------|-------------------------------|
| `QualityAPISR` | `quality` | ~5.9M | Plain 3×3 conv | ~15-25 FPS |
| `MidAPISR` | `mid` | ~3.0M | Plain 3×3 conv (narrower) | ~30-50 FPS |
| `BSConvAPISR` | `bsconv` | ~0.4M | Blueprint Separable Conv | ~50-80 FPS |
| `FastAPISR` | `fast` | ~0.2M | Channel-split depthwise | ~80-120 FPS |

All variants share the same head→residual body→PixelShuffle upsample→clamp structure. The scale factor (2x or 4x) is a constructor argument and controls the upsample stage: 4x uses two cascaded 2x PixelShuffles to avoid checkerboard artifacts on anime line art; 2x uses a single stage.

Model I/O contract: NCHW float32, values in `[0, 1]`, no mean/std normalization. Output is clamped to `[0, 1]` by the model's own `forward()`.

### Training

The loss function (`TwinPerceptualLoss`) uses a warmup schedule: pure L1 for the first N epochs (sharp edges first), then gradually ramps in a VGG19+ResNet18 perceptual loss (texture quality). This two-phase approach avoids the blurriness that perceptual loss causes when applied from the start.

The dataset (`SRDataset`) randomly crops HR patches from your image directory, applies spatial augmentations (flip, rotate) and mild color jitter, then downsamples to LR using Lanczos — producing (LR, HR) training pairs on the fly.

### The DeepStream pipeline

The full GStreamer pipeline when upscaling is active:

```
filesrc → parsebin → nvv4l2decoder (NVDEC) → nvstreammux
→ nvvideoconvert → [caps filter: NVMM RGBA]
→ nvdsvideotemplate (custom APISR plugin)
→ nvvideoconvert → nv3dsink
```

Key design decisions, each driven by a real bug encountered during development:

**`parsebin` instead of `decodebin`**: `decodebin`'s autoplugger fails to negotiate NVDEC for 10-bit HEVC (Main10 profile) even though the explicit `demux → parse → nvv4l2decoder` chain works perfectly. `parsebin` handles demuxing and bitstream parsing without trying to autopick the final decoder, letting us wire `nvv4l2decoder` in ourselves.

**Explicit NVMM caps filter**: Without `video/x-raw(memory:NVMM),format=RGBA` between `nvvideoconvert` and `nvdsvideotemplate`, the converter silently drops the `memory:NVMM` caps feature. The plugin then receives plain system-memory buffers instead of `NvBufSurface` pointers, and casting raw pixel bytes to the `NvBufSurface` struct produces garbage fields that cause an immediate SIGSEGV. Confirmed via `gst-launch -v` caps tracing.

**Synchronous `ProcessBuffer`** (no `OutputThread`): The async queue + background thread pattern from NVIDIA's reference implementation causes a deadlock with real video decoders. The streaming thread blocks inside `nvv4l2decoder` waiting for a V4L2 capture buffer to be returned, while `OutputThread` blocks inside `gst_pad_push` waiting for the streaming thread to process serialized events. Four frames process successfully (matching the decoder's internal capture buffer count), then everything freezes. Processing synchronously on the streaming thread avoids this entirely, and at 8-15ms/frame is easily fast enough for real-time 30fps.

**Engine pre-load in `SetProperty`**: `GetCompatibleCaps` fires during caps negotiation, before `SetInitParams` loads the engine. If the scale factor defaults to 1 at that point, the output buffer pool gets allocated at input resolution — the subsequent upscaled output then writes past the end of the buffer. Loading the engine in `SetProperty` (when `engine-path` is set) ensures `m_scaleFactor` is correct before negotiation happens.

**First-audio-only gating**: Files with multiple audio tracks (common in dual-audio anime rips) create multiple `autoaudiosink` instances that compete for the PulseAudio clock, producing ever-growing clock skew that stalls the pipeline. Only the first audio track is routed to a real sink; additional tracks drain to `fakesink`.

### The custom plugin

The plugin (`libcustom_videoimpl.so`) implements the `IDSCustomLibrary` interface from `nvdsvideotemplate`. Its `ProcessBuffer` method runs the full SR pipeline on each incoming frame:

1. **RGBA → planar float32**: NPP `nppiCopy_8u_C4P4R` splits RGBA into 4 uint8 planes (alpha discarded), then `nppiConvert_8u32f` + `nppiMulC` normalize each channel to `[0, 1]`.

2. **Tile extraction**: A custom CUDA kernel (`launchExtractTiles`) copies all tiles from the planar frame into the TRT engine's batched input buffer in one parallel launch — one thread per (tile, channel, row, column). This replaced an earlier per-tile NPP loop that issued ~700 individual kernel launches per frame.

3. **TRT inference**: The engine runs all tiles in the batch. With 96 tiles at maxBatch=64, this is 2 calls per frame.

4. **Cosine-blend stitch**: Another custom kernel (`launchStitchTiles`) accumulates all output tiles into the upscaled frame using `atomicAdd` with cosine-weighted blending, matching the `_make_blend_mask` function from `train.py`/`quant.py`. A separate weight accumulator tracks total blend weight per pixel.

5. **Normalize + pack**: NPP divides the accumulator by the weight buffer, scales back to `[0, 255]`, converts to uint8, then a final custom kernel (`launchPackRgbFromPlanes`) interleaves the 3 uint8 channel planes into RGBA. This last kernel exists because no NPP function performs a "3 planes in, skip the 4th channel out" interleave — confirmed by grepping the real installed `npp.h` headers.

### Runtime scale detection

The plugin reads the engine's optimization profile at load time to derive scale, tile sizes, and max batch — so swapping between a 2x and 4x engine requires no code changes or recompilation. The engine's input opt shape gives the LR tile size, `setInputShape` + `getTensorShape` on the output gives the HR tile size (since `getProfileShape` is input-only in TRT 10.x), and the ratio determines the scale. This is logged on startup:

```
SrTrtEngine: loaded apisr_fp16.trt
  scale=2  LR tile=64  HR tile=128  maxBatch=64
```


## Setup

### Prerequisites

- Ubuntu 24.04 (tested; other Linux should work with path adjustments)
- NVIDIA GPU with Compute Capability ≥ 7.0 (Turing or newer)
- NVIDIA driver ≥ 550
- CUDA toolkit (tested with 13.2)
- DeepStream 9.0 (installed from the `.deb`, not apt — it's not in NVIDIA's public apt repo)
- TensorRT 10.16 (the version DeepStream 9.0 ships with — **not** 11.x)
- Python 3.12 with system-installed GObject introspection bindings

### TensorRT version pinning

DeepStream 9.0 links against TensorRT 10.16, but NVIDIA's apt repo defaults to installing the latest TRT (11.x). If left unpinned, `libnvinfer-dev`, `libnvinfer-bin`, and related packages drift to 11.x, causing serialization mismatches when loading engines. Pin them:

```bash
sudo apt install \
  libnvinfer-bin=10.16.1.11-1+cuda13.2 \
  libnvinfer-dev=10.16.1.11-1+cuda13.2 \
  libnvinfer-headers-dev=10.16.1.11-1+cuda13.2 \
  libnvinfer-headers-plugin-dev=10.16.1.11-1+cuda13.2 \
  libnvinfer-safe-headers-dev=10.16.1.11-1+cuda13.2 \
  libnvinfer-plugin-dev=10.16.1.11-1+cuda13.2 \
  libnvonnxparsers-dev=10.16.1.11-1+cuda13.2

sudo apt-mark hold libnvinfer-bin libnvinfer-dev libnvinfer-headers-dev \
  libnvinfer-headers-plugin-dev libnvinfer-safe-headers-dev \
  libnvinfer-plugin-dev libnvonnxparsers-dev
```

### Python dependencies (training/benchmarking)

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision onnx onnxruntime-gpu numpy pillow tqdm tensorrt
```

**Important**: The venv's `tensorrt` package version will almost certainly differ from the system TRT that DeepStream uses. This is fine for training, ONNX export, and Python-side benchmarking — but **never use `python quant.py export` to build engines the C++ plugin will load**. Always use the system `trtexec` binary for engine builds (see below). Deactivate the venv before building engines or running the player.

### System dependencies (player)

```bash
sudo apt install python3-gi gir1.2-gtk-3.0 gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0
```

These are GObject introspection bindings for GTK and GStreamer — they install into the system Python, not a venv. Run `player.py` with the system Python, not inside a venv.


## Usage

### 1. Train a model

```bash
source venv/bin/activate

# 2x upscale, BSConv variant (recommended starting point)
python train.py \
  --data_dir /path/to/anime/images \
  --sr_rate 2 \
  --model bsconv \
  --hr_size 128 \
  --epochs 100 \
  --save_path model/apisr_2x.pth

# 4x upscale, quality variant (slower, higher quality)
python train.py \
  --data_dir /path/to/anime/images \
  --sr_rate 4 \
  --model quality \
  --hr_size 192 \
  --epochs 100 \
  --save_path model/apisr_4x.pth
```

Training automatically exports an ONNX file alongside the `.pth` checkpoint and runs a final evaluation with PSNR numbers.

`--hr_size` should be divisible by `--sr_rate`. For 2x, `128` means 64×64 LR tiles → 128×128 HR tiles. For 4x, `192` means 48×48 LR → 192×192 HR.

Other useful args: `--resume model/apisr_epoch50.pth` to continue from a checkpoint, `--model fast` for the smallest/fastest variant, `--downsample lanczos` (default) or `bicubic`.

### 2. Build the TensorRT engine

**Always deactivate the venv first.** The system `trtexec` (TRT 10.16, matching DeepStream) must be used, not the venv's Python TRT package.

```bash
deactivate

/usr/bin/trtexec \
  --onnx=model/apisr_2x.onnx \
  --saveEngine=model/trt_engines/apisr_2x_fp16.trt \
  --fp16 \
  --minShapes=input:1x3x32x32 \
  --optShapes=input:64x3x64x64 \
  --maxShapes=input:64x3x128x128
```

The shape profile `(batch, channels, height, width)` controls what the engine can handle at runtime. `optShapes` is what TRT tunes for (set batch to your typical tiles-per-call); `maxShapes` is the ceiling (tiles above this fail). The spatial dimensions (64×64 for opt, 128×128 for max) are the LR tile size, not the output.

First build takes 1-5 minutes as TRT benchmarks hundreds of kernel implementations for your specific GPU.

### 3. Benchmark (optional)

```bash
source venv/bin/activate

python quant.py bench \
  --onnx model/apisr_2x.onnx \
  --trt_dir model/trt_engines/ \
  --data_dir /path/to/anime/images \
  --sr_rate 2 \
  --batch_sz 16
```

Prints a table comparing ORT FP32 vs TRT FP32 vs TRT FP16 in FPS and PSNR.

### 4. Build the DeepStream plugin

```bash
deactivate  # must be outside the venv

cd plugin/
CUDA_VER=13.2 make clean && CUDA_VER=13.2 make
sudo CUDA_VER=13.2 make install
```

This compiles `libcustom_videoimpl.so` and installs it to `/opt/nvidia/deepstream/deepstream-9.0/lib/`. The Makefile targets `sm_86` (RTX 3080 / Ampere) — change the `-arch` flag if your GPU has a different compute capability.

Verify the build:
```bash
nm -D libcustom_videoimpl.so | grep CreateCustomAlgoCtx
# should print a symbol — this is what DeepStream dlopen's
```

### 5. Run the player

```bash
deactivate  # must be outside the venv — player uses system Python + GI bindings

cd player/
python3 player.py
```

Configure the three constants at the top of `player.py`:

```python
CUSTOMLIB_PATH = "/opt/nvidia/deepstream/deepstream-9.0/lib/libcustom_videoimpl.so"
ENGINE_PATH    = "/absolute/path/to/model/trt_engines/apisr_2x_fp16.trt"
OVERLAP_HR     = 4    # HR pixels; lower = fewer tiles = faster, at slight seam risk
```

Click "Open Video...", pick a file. The upscaled video plays in DeepStream's own EGL window (`nv3dsink`), not embedded in the GTK window. Audio plays through PulseAudio. Seeking is supported via the slider.

Set `CUSTOMLIB_PATH = None` to run in decode-only mode (no upscaling) for comparison.

### Standalone test (no GUI)

```bash
# H.264 file
gst-launch-1.0 filesrc location="video.mkv" ! matroskademux ! h264parse ! nvv4l2decoder ! \
  mux.sink_0 nvstreammux name=mux batch-size=1 width=720 height=480 ! \
  nvvideoconvert ! "video/x-raw(memory:NVMM),format=RGBA" ! \
  nvdsvideotemplate customlib-name="/opt/nvidia/deepstream/deepstream-9.0/lib/libcustom_videoimpl.so" \
    customlib-props="engine-path:/absolute/path/to/apisr_fp16.trt" \
  ! nvvideoconvert ! nv3dsink

# H.265 / HEVC file (including 10-bit Main10)
gst-launch-1.0 filesrc location="video.mkv" ! matroskademux ! h265parse ! nvv4l2decoder ! \
  mux.sink_0 nvstreammux name=mux batch-size=1 width=1440 height=1080 ! \
  nvvideoconvert ! "video/x-raw(memory:NVMM),format=RGBA" ! \
  nvdsvideotemplate customlib-name="/opt/nvidia/deepstream/deepstream-9.0/lib/libcustom_videoimpl.so" \
    customlib-props="engine-path:/absolute/path/to/apisr_fp16.trt" \
  ! nvvideoconvert ! nv3dsink
```

Adjust `h264parse`/`h265parse` and `width`/`height` to match your file. Use `gst-discoverer-1.0 video.mkv` to check codec and resolution.


## Tuning

### Overlap

`OVERLAP_HR` controls how many HR pixels adjacent tiles share. Higher overlap means more tiles per frame (slower) but smoother blending at tile boundaries. Lower overlap means fewer tiles (faster) but potential visible seams if the model's receptive field doesn't cover the gap.

| OVERLAP_HR | Tiles for 720×480 2x | Notes |
|-----------|---------------------|-------|
| 16 | ~150 | Conservative, no visible seams |
| 8 | ~96 | Good balance for most content |
| 4 | ~80 | Fastest, seams unlikely at 480p |

### Scale factor

2x is recommended for real-time playback. 4x produces beautiful results but generates 4× the output pixels and roughly 4× the tiles, which is too slow for real-time on most content without a very fast model variant.

To switch scales: train a new model with `--sr_rate 2` or `4`, export ONNX, build a TRT engine, and point `ENGINE_PATH` at it. The plugin auto-detects the scale from the engine — no code changes or recompilation needed.

### Batch size

The engine's `--maxShapes` batch dimension caps how many tiles go into one TRT call. For 96 tiles at maxBatch=64, that's 2 TRT calls per frame. At maxBatch=128, it's 1 call. Larger batches amortize per-call overhead but use more GPU memory for TRT's internal scratch. Watch `nvidia-smi` during the engine build — the RTX 3080's 12GB is the practical ceiling.


## Debugging

### `deepstream-app --version-all`

Should print clean with TRT 10.16, cuDNN 9.x, no `Could not load library` errors. If it crashes loading cuDNN, check for stale `/etc/ld.so.conf.d/` entries or `LD_LIBRARY_PATH` pointing at old CUDA installations.

### `gst-launch-1.0 -v`

The `-v` flag prints negotiated caps on every pad. Check that both `nvdsvideotemplate0.sink` and `nvdsvideotemplate0.src` show `video/x-raw(memory:NVMM)` — if the sink side is missing `(memory:NVMM)`, the caps filter is not working and the plugin will crash.

### `compute-sanitizer`

```bash
compute-sanitizer --tool memcheck gst-launch-1.0 ... 2>&1 | grep "Invalid"
```

Catches out-of-bounds GPU memory accesses. The most common cause during development was output buffers allocated at input resolution (scale=1) due to the engine not being loaded before caps negotiation.

### `gdb`

```bash
gdb --args gst-launch-1.0 ...
(gdb) run
(gdb) bt
```

GStreamer catches SIGSEGV and spins instead of dying, so you can also attach to a running process when it prints "Spinning. Please run gdb ...".

### Common warnings

`gst_caps_set_simple: assertion 'IS_WRITABLE (caps)' failed` — cosmetic, non-fatal, inherent to `gstnvdsvideotemplate`'s own caps handling. Does not cause any functional issue.

`Gst.Element.get_request_pad is deprecated` — use `request_pad_simple` instead. Non-breaking.


## Known limitations

- **Memory at 4x on large inputs**: A 1440×1080 source at 4x produces 5760×4320 output. Combined with TRT's internal scratch memory (~4-8GB depending on engine profile), this can exceed the RTX 3080's 12GB. The 2x model is the practical choice for sources above ~720p.

- **First-frame warmup**: The first frame after pipeline start shows as white (decoder warm-up). This is normal NVDEC behavior, not an SR bug — the same thing happens with no upscaler in the pipeline.

- **No inter-frame pipelining**: Each frame is fully processed (decode → tile → TRT → blend → display) before the next starts. Overlapping frame N+1's decode with frame N's inference would improve throughput further but requires a second CUDA stream and careful buffer management.

- **Pixel aspect ratio**: Anamorphic content (non-square pixels, e.g. 720×480 with PAR 8/9) is upscaled at stored dimensions, not display dimensions. The output may appear slightly stretched compared to a PAR-aware player.
