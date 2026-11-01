# Optional SAM 3.1 and YOLOX fine-tuning packages

## Windows: SAM 3.1 road-mask add-on

Install RoadLoom first. On a Windows PC with internet access, close RoadLoom and run `scripts\\Setup-SAM-Road-Masks.cmd` from the installed application folder (normally `%LOCALAPPDATA%\\Programs\\RoadTrafficAnalysis`). It downloads the public BF16 SAM 3.1 model and installs its separate CPU runtime under the current user's app data. No Hugging Face account, sign-in, or token is used. The model is about 1.68 GB; leave several GB free for the runtime and temporary setup files. After successful setup, the app loads the local copy and can be used without internet. To use a fully disconnected Windows PC, prepare the model/runtime on a connected Windows PC and copy the complete `sam3.1-unsloth` folder to `%LOCALAPPDATA%\\RoadTrafficAnalysis\\Models\\sam3.1-unsloth` on the offline PC.

This model suggests road masks for operator review; it does not fine-tune SAM and does not replace vehicle detection. SAM execution is CPU-only in the current RoadLoom adapter.

## Windows: CUDA fine-tuning add-on

The separate CUDA bundle fine-tunes the YOLOX-S vehicle detector; it does **not** train SAM. Build it on a Windows x64 staging PC with a prepared portable Python 3.11/3.12 runtime, CUDA-enabled PyTorch, OpenCV, ONNX, and a local YOLOX 0.3.0 checkout. Use the command in [deployment and fine-tuning instructions](07-deployment.md#optional-offline-yolox-fine-tuning-bundle). Add `--require-cuda` when the staging PC has the target NVIDIA GPU and you want the bundle preflight to run a CUDA operation there. The resulting ZIP includes the runtime, trainer, YOLOX source, baseline weights, and hashes; it does not include reviewed training images. Run `verify-training.cmd` on the offline PC before training.

The current workspace is macOS and has no NVIDIA CUDA device, so a CUDA bundle cannot be built or hardware-validated here. The Windows bundle must be staged on Windows; target GPU compatibility must be checked on the intended workstation.

## macOS

Use [the macOS preview setup](13-macos-preview.md). The app and SAM 3.1 road-mask add-on run locally on Mac after their one-time setup downloads. CUDA is unsupported on macOS. PyTorch MPS on Apple Silicon is a possible later YOLOX training backend, but RoadLoom's current fine-tuning path requires CUDA and has not been ported or verified for MPS.
