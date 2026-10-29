# Optional SAM 3.1 road-mask assistant

The core installer does not include this optional segmentation runtime or weights. The app runs SAM independently on the selected image with the text prompt `road surface`; it returns candidate masks for you to inspect. Choosing a mask only fills the editable polygon draft. You still choose its purpose and save it. A saved analysis-area polygon masks pixels outside that region before YOLO/light inference. A road-surface polygon only adds an on/near/off-road review cue. SAM does not replace or disable YOLO vehicle detection.

To enable the button, install a local Transformers build that provides `Sam3Model` and `Sam3Processor`, PyTorch, and Pillow. Copy the model files from the Unsloth AI Transformers-format SAM 3.1 checkpoint into this directory, including `config.json`, `model.safetensors`, processor/tokenizer configuration, and vocabulary files. The standard BF16 checkpoint is about 1.68 GB; the optional INT8 checkpoint is about 992 MB, but this app currently loads the standard `model.safetensors` file on CPU. Do not rename the INT8 file to pretend it is the standard checkpoint. The app never downloads models or accesses Hugging Face at run time.

The model is run in float32 on CPU for broad operator compatibility. This uses several GB of working memory and can be slow. It has not yet been benchmarked on the target Windows workstation. Set `ROAD_TRAFFIC_SAM3_DIR` to a local model directory if it is stored elsewhere.

Sources:
- https://huggingface.co/unsloth/sam3.1 (community Transformers-format model and quantized variants)
- https://github.com/facebookresearch/sam3 (upstream SAM 3.1 model code)
