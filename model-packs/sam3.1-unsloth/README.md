# Optional SAM 3.1 road-mask assistant

The core installer does not include this optional segmentation runtime or weights. The app runs SAM independently on the selected image with the text prompt `road surface`; it returns candidate masks for you to inspect. Choosing a mask only fills the editable polygon draft. You still choose its purpose and save it. A saved analysis-area polygon masks pixels outside that region before YOLO/light inference. A road-surface polygon only adds an on/near/off-road review cue. SAM does not replace or disable YOLO vehicle detection.

No Hugging Face account, sign-in, or token is needed. The Unsloth SAM 3.1 model is public; download its files anonymously, then install the local Transformers runtime and copy the model files here. Include `config.json`, BF16 `model.safetensors` (about 1.68 GB), processor/tokenizer configuration, and vocabulary files. Do not rename a quantized file to pretend it is the standard checkpoint. Once files are local, the app runs offline and never contacts Hugging Face.

The app currently requires the full BF16 `model.safetensors` file and loads it in float32 on CPU for broad operator compatibility. It does not load the smaller INT8/FP8 files: those are pre-quantized weights that require their own quantized inference path, and simply renaming them or passing them to the standard loader will not work. The BF16 model uses several GB of working memory and can be slow. Performance has not been benchmarked on the target Windows workstation. Set `ROAD_TRAFFIC_SAM3_DIR` to a local model directory if it is stored elsewhere. SAM and its model files remain separate from the main installer.

Sources:
- https://huggingface.co/unsloth/sam3.1 (community Transformers-format model and quantized variants)
- https://github.com/facebookresearch/sam3 (upstream SAM 3.1 model code)
