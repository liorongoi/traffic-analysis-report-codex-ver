# Fine-tune the vehicle detector

RoadLoom includes a COCO-pretrained YOLOX-S PyTorch baseline and can export an evidence-versioned COCO ZIP from complete human reviews. The optional trainer starts from that local baseline, uses only the training and validation roles, and leaves the held-out test split unused. It writes checkpoints, training data and logs to the output folder, and puts the fixed-size ONNX model plus manifest in a dedicated `roadloom-model-pack` subfolder. The training profile currently requires one CUDA-capable NVIDIA GPU and a separate PyTorch/YOLOX training environment; none of these packages or large training dependencies are added to the main installer. No Hugging Face account, sign-in, or token is used.

After importing a trained pack, evaluate both the existing baseline and the candidate against the same frozen, independently labeled test split. The offline report comparator in `docs/06-benchmarks.md` highlights changes by class and available test strata. This is evidence for operator review; it has no built-in acceptance threshold and does not change the selected mission model. Representative test data and product-approved thresholds are still needed before accuracy can be accepted.

## Prepare the labels

1. In the app's learning panel, assign reviewed images to training, validation or test. Review each training and validation image completely; partial reviews are excluded. Keep related camera/hour captures in one split.
2. Download **Export reviewed learning dataset** from the same panel.
3. Prepare Windows x64 Python 3.11 or 3.12, CUDA-enabled PyTorch 2.5.1/torchvision 0.20.1, and official YOLOX 0.3.0 in a separate environment. The offline bundle stager accepts only Python 3.11 and 3.12. The YOLOX source tree can also be used with `--yolox-root`. Install the YOLOX runtime's remaining requirements there, including ONNX and COCO evaluation dependencies. Use YOLOX's Windows JIT build requirements when training on Windows.
4. From the repository root, run:

```powershell
python scripts/finetune-yolox.py `
  --dataset-zip C:\Data\RoadLoom-Learning-Dataset.zip `
  --output C:\Data\RoadLoom-YOLOX-Candidate `
  --epochs 50 `
  --batch-size 4
```

The script checks the bundled baseline checksum, stages only explicitly reviewed vehicle boxes for the supported classes (bicycle, car, motorcycle, bus, truck), preserves complete reviewed images with no vehicle box as negative examples, and ignores non-vehicle annotations. It fails if either training or validation has no usable vehicle data, and rejects unsafe image paths. Training image size is 640×640. The default batch size is deliberately small; adjust it to fit GPU memory.

The app's Vehicle detector model panel can import the `roadloom-model-pack` folder directly. It copies the ONNX model and manifest into the local candidate registry, where an operator can choose it for future still-image and video runs and record the reason. Each run freezes its model ID and hash; resuming a run or repeating a changed-region run uses the same frozen model. Video continuous intake uses the mission selection when each source run is created. No candidate is selected automatically.

Training uses validation metrics to choose `best_ckpt.pth`; it never passes the held-out test annotations to YOLOX. The current model selection screen records a mission-local operator choice but does not yet gate it on a held-out detector evaluation. Compare the candidate against the untouched test role and representative camera/night conditions before relying on it. Numerical acceptance thresholds and a real target-GPU training run remain open.

## Upstream references

- [YOLOX custom-data training guide](https://yolox.readthedocs.io/en/latest/train_custom_data.html)
- [YOLOX 0.3.0 release notes](https://github.com/Megvii-BaseDetection/YOLOX/releases)
- [YOLOX Windows/JIT installation code](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/setup.py)
