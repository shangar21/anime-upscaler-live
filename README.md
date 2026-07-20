# Anime Super-Resolution Video Player

This software is a real-time anime video upscaler that operates on the GPU. It decodes a video file, increases the resolution of every frame, and shows the result. It uses a trained super-resolution model that operates on TensorRT. All processing occurs on the GPU. The software does not copy memory between the CPU and the GPU.  

The software uses DeepStream 9.0 and GStreamer for the media pipeline. A custom nvdsvideotemplate plugin operates the TensorRT engine for inference. The graphical user interface (GUI) uses GTK to let you select files and play video.  

We tested this software on Ubuntu 24.04 with an RTX 3080 (12GB) GPU. It operates at 30 frames per second (FPS) on 720×480 anime video with the 2x model.  

<img width="1916" height="700" alt="live_upscaler drawio(2)" src="https://github.com/user-attachments/assets/939b301e-97a5-4067-93e0-e96a1f520e6a" />

## Demo



https://github.com/user-attachments/assets/9c983ae0-6e14-4ec8-93cc-3d9f13eef664



## Project structure

```
.
├── model/
│   ├── apisr.py              # Contains four model architectures
│   ├── train.py              # Controls the training loop, ONNX export, and evaluation
│   ├── dataset.py            # Creates the dataset and applies augmentation
│   ├── quant.py              # Builds the TensorRT engine and runs benchmarks
│   └── trt_engines/          # Contains the compiled .trt engine files
│
├── player/
│   ├── customlib_impl.cpp    # Contains the custom DeepStream plugin
│   ├── sr_trt_engine.hpp/cpp # Loads the TensorRT engine and detects the scale
│   ├── sr_pipeline.hpp/cpp   # Controls the tiling and blend pipeline
│   ├── tile_kernels.cuh/cu   # Contains CUDA kernels to extract and stitch tiles
│   ├── pack_rgba_kernel.cuh/cu  # Contains CUDA kernels to combine RGB planes into RGBA
│   └── Makefile              # Compiles libcustom_videoimpl.so
│   └── player.py             # Contains the GTK GUI
│
├── .gitignore
└── README.md
```


## How it works

### The model

APISR is a neural network. It increases the resolution of a single image. We trained it on anime video. It receives a low-resolution RGB image area, such as 64×64 pixels. It outputs a high-resolution image area. The output is 128×128 pixels for 2x scale, or 256×256 pixels for 4x scale. The model is a residual network that uses a PixelShuffle upsampler. There are four versions of the model. These versions balance visual quality and processing speed:

| Variant | Tag | Params | Block type | Expected speed (720p, TRT FP16) |
|---------|-----|--------|------------|-------------------------------|
| `QualityAPISR` | `quality` | ~5.9M | Plain 3×3 conv | ~15-25 FPS |
| `MidAPISR` | `mid` | ~3.0M | Plain 3×3 conv (narrower) | ~30-50 FPS |
| `BSConvAPISR` | `bsconv` | ~0.4M | Blueprint Separable Conv | ~50-80 FPS |
| `FastAPISR` | `fast` | ~0.2M | Channel-split depthwise | ~80-120 FPS |

All model versions use the same structure: head, residual body, PixelShuffle upsample, and clamp. The scale factor (2x or 4x) controls the upsample stage. The 4x scale uses two 2x PixelShuffle stages. This prevents checkerboard artifacts on anime line art. The 2x scale uses one stage.  

The model requires NCHW float32 input. The values must be between [0, 1]. Do not use mean or standard deviation normalization. The forward() function limits the output to [0, 1].  

Model I/O contract: NCHW float32, values in `[0, 1]`, no mean/std normalization. Output is clamped to `[0, 1]` by the model's own `forward()`.

### Training

The loss function (TwinPerceptualLoss) uses a two-phase procedure. First, it uses L1 loss for the initial epochs to make sharp edges. Second, it adds a VGG19 and ResNet18 perceptual loss to improve texture quality. This procedure prevents blurry images.  

The dataset (SRDataset) cuts high-resolution areas from your images. It changes the spatial orientation and colors. Then, it decreases the resolution with a Lanczos filter. This creates pairs of low-resolution and high-resolution images for training.  

### The DeepStream pipeline

The full GStreamer pipeline when upscaling is active:

```
filesrc → parsebin → nvv4l2decoder (NVDEC) → nvstreammux
→ nvvideoconvert → [caps filter: NVMM RGBA]
→ nvdsvideotemplate (custom APISR plugin)
→ nvvideoconvert → nv3dsink
```

These are the primary design decisions:  
- parsebin instead of decodebin: The decodebin function cannot use NVDEC for 10-bit HEVC video. The parsebin function separates the video stream. It lets us connect nvv4l2decoder manually.
- Explicit NVMM caps filter: You must put video/x-raw(memory:NVMM),format=RGBA between nvvideoconvert and nvdsvideotemplate. If you do not, the converter removes the memory:NVMM property. The plugin will then receive system-memory buffers instead of NvBufSurface pointers. This causes a SIGSEGV error.
- Synchronous ProcessBuffer: Do not use an asynchronous queue and background thread. This causes a deadlock error with video decoders. Synchronous processing on the streaming thread prevents this error. It requires 8 to 15 milliseconds per frame. This is sufficiently fast for 30 FPS video.
- Engine pre-load in SetProperty: You must load the engine in SetProperty when you set engine-path. This makes sure m_scaleFactor is correct before the software allocates the output buffer. If the scale factor is incorrect, the output writes to invalid memory.
- First-audio-only gating: Videos with multiple audio tracks cause synchronization errors. The software sends only the first audio track to the output. It sends the other tracks to fakesink

### The custom plugin
The custom plugin (libcustom_videoimpl.so) uses the IDSCustomLibrary interface. The ProcessBuffer function operates the super-resolution pipeline on every frame:  
1. RGBA to planar float32: The software separates RGBA data into 4 uint8 planes. It discards the alpha plane. It changes the data to float32 and scales it to [0, 1].
2. Tile extraction: A custom CUDA kernel (launchExtractTiles) copies all tiles into the input buffer. This occurs in one parallel operation.
3. TensorRT inference: The engine processes all tiles in the batch.  Cosine-blend stitch: A custom CUDA kernel (launchStitchTiles) puts the output tiles into the high-resolution frame. It uses cosine-weighted blending.
4. Normalize and pack: The software normalizes the data and changes it to uint8. A custom CUDA kernel (launchPackRgbFromPlanes) combines the 3 uint8 planes into RGBA format. 

### Runtime scale detection

The plugin reads the engine profile when it loads. It finds the scale, tile sizes, and maximum batch size. You do not have to change the code or compile the software to change between 2x and 4x models. The software logs this data when it starts:  

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

DeepStream 9.0 requires TensorRT 10.16. The default NVIDIA repository installs TensorRT 11.x. You must pin the TensorRT packages to version 10.16. If you do not, the engine files will not load. Use these commands:

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

Important: The TensorRT version in the virtual environment is different from the system version. This is acceptable for training. Do not use python quant.py export to build the engine files for the C++ plugin. You must use the system trtexec file. Deactivate the virtual environment before you build engines or operate the player.

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
