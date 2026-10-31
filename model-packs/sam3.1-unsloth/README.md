# Optional SAM 3.1 road-mask assistant

The core installer does not include this optional segmentation runtime or weights. The app runs SAM independently on the selected image with the text prompt `road surface`; it returns candidate masks for you to inspect. Choosing a mask only fills the editable polygon draft. You still choose its purpose and save it. A saved analysis-area polygon masks pixels outside that region before YOLO/light inference. A road-surface polygon only adds an on/near/off-road review cue. SAM does not replace or disable YOLO vehicle detection.

On Windows, install RoadLoom first, close it, connect to the internet, then run `scripts\Setup-SAM-Road-Masks.cmd` from the installed RoadLoom folder. This installs the optional CPU PyTorch/Transformers runtime beside the weights in the model folder and downloads the public BF16 model to `%LOCALAPPDATA%\RoadTrafficAnalysis\Models\sam3.1-unsloth`. It does not change or rebuild the single core installer; the optional setup leaves the core app runtime unchanged. It requires Python 3.11+ in the app runtime and several GB of temporary and installed disk space. The setup is one-time; close and reopen RoadLoom afterward. For an offline workstation, run setup on the intended machine while connected, then move the machine offline. Application use after setup makes no Hugging Face requests.

No Hugging Face account, sign-in, or token is used. The model is downloaded from the public `unsloth/sam3.1` Transformers-format repository with `token=False`; no cached credentials are read. The existing downloader excludes FP8/INT8 checkpoints and validates that the complete BF16 `model.safetensors` file is present. The current app loads BF16 weights in float32 on CPU for compatibility. This can use several GB of memory and be slow; target Windows performance is unverified. Keep the model folder local. It is outside Git and outside the core installer.

The separate `scripts/setup-sam3.1.py` command can take `--destination PATH` to select another model folder. Set `ROAD_TRAFFIC_SAM3_DIR` to the same folder before launching RoadLoom. Do not rename a quantized file to pretend it is the standard checkpoint.

Sources:
- https://huggingface.co/unsloth/sam3.1 (public Transformers-format SAM 3.1 model and checkpoint variants)
- https://huggingface.co/docs/transformers/model_doc/sam3 (Transformers SAM 3 runtime/API)
