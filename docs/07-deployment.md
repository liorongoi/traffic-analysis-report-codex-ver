# 07 — Hardware profiles and offline deployment

Target: Windows workstation, 16/48 GB RAM, RTX A4000 or other graphics, one active user. Development execution on this Mac does not establish Windows or GPU compatibility.

Windows inference now selects ONNX Runtime DirectML automatically when available and falls back to CPU if setup or inference fails. DirectML sessions are serialized and configured for its memory-pattern and execution-mode requirements. The offline Windows runtime pins `onnxruntime-directml`; the macOS runtime remains CPU-only. The on-demand host report lists providers without running a model, so GPU operation and performance on the target workstation still need native validation. RAM profiles remain advisory and set CPU thread limits; inference still uses one worker. Avoid eager model loading and simultaneous inference/training memory contention.

## Packaging design

Create immutable, versioned prerequisite/runtime/model packs plus small application updates. Stage pinned wheels, native libraries, local models, codecs, timezone data/boundaries, fonts and scripts on an internet-connected build machine. Record software/weights/data licenses separately and create a content manifest. Include all actual required assets in copy-ready media. No package managers/model pulls/CDNs/telemetry during installed operation or installation.

Installation and normal operation must not need administrator privileges. Use %LOCALAPPDATA%, current-user shortcuts and private runtimes. Local folders and authorized UNC shares are required. Autostart and credential handling remain design/testing work. Do not install a system service/global runtime. The preview binds to loopback by default. Optional LAN mode exists in the code but LAN acceptance is out of scope for this delivery. There is no app-level sign-in or account; one operator uses the local app. Missing model packs must fail clearly. If DirectML is unavailable or fails, inference uses the pinned CPU provider; models are never silently downloaded or substituted.

Installation performs resumable content/native checks; normal launch reuses a receipt bound to exact content. The workspace and mission data are stored outside the installed application directory. CI verifies that reinstalling the current installer preserves a workspace sentinel; rollback from an earlier installer release remains untested. Validate actual EXE contents and copied-file hashes independently, then verify native Windows offline installation/update/rollback on target hardware.

The current staging source rejects non-Windows/non-x64 Python executables, checks installed packages against `requirements-inference-windows.lock.txt`, requires offline tzdata, and verifies the vehicle and plate-region detector packs. OCR models and text-review assets are not staged or required. The payload includes both detector packs, worker entry points, web assets, and deployment documentation. The install verifier checks copied-file hashes, runs both bundled models on a synthetic blank frame, records the selected ONNX providers, fallback reasons, model initialization and inference timings, host memory/GPU/storage inventory, and writes a manifest-bound receipt. It blocks Python socket connections during the smoke check; this is not an OS-level network-isolation test. Normal launch checks the receipt.

### Offline hardware compatibility report

Installation writes `inference-runtime-validation.txt` beside the installed app. On the target Windows PC, inspect that JSON-formatted report for the CPU/RAM/GPU inventory, DirectML or CPU provider actually selected by each model, fallback reasons, and synthetic-frame timings. For additional measurements using representative still images while the computer is disconnected, run:

```powershell
$root = Join-Path $env:LOCALAPPDATA 'Programs\RoadTrafficAnalysis'
& "$root\runtime\python.exe" -I "$root\scripts\check-inference-runtime.py" `
  "$root" --images 'D:\TrafficSamples\still-images' --max-images 8 `
  --output "$env:USERPROFILE\Desktop\roadloom-offline-report.json"
```

This reads images without using their capture timestamps and reports image hashes, dimensions, vehicle/plate-region proposal counts, and per-model timing percentiles. It does not score detection accuracy, read plate text, or analyze video. The current acceptance target is successful offline execution, compatible selected providers, and an inspectable report; no numerical speed or accuracy threshold is invented. The report allows optimization to target measured bottlenecks. The Windows installer workflow checks the report survives install and reinstall.

Windows run [41192125897](https://github.com/liorongoi/traffic-analysis-report-codex-ver/actions/runs/41192125897) on commit 14f2d2106cebd5b8d6eaa9211a1f825f6b71d906 passed all **365 tests in 199.181 seconds**, staged the offline application payload, compiled the installer, verified install/reinstall/uninstall with workspace preservation, launched the installed app and confirmed its local RoadLoom page returned HTTP 200, reinstalled while it was running and verified that Setup closed it, seeded five obsolete OCR model files and verified Setup removed them, and uploaded artifact [13039786203](https://github.com/liorongoi/traffic-analysis-report-codex-ver/actions/runs/41192125897/artifacts/13039786203) (258,226,100 bytes; SHA-256 `ee4da785446ccc52c41be5f45aaafa7e07952f787ad6d28fcf37b90bc7d51a10`; expires 2026-11-07 10:25 UTC). This verifies core installer behavior in hosted Windows CI; real SAM inference and target-machine memory/runtime performance remain unverified.

The latest acceptance run [41277874562](https://github.com/liorongoi/traffic-analysis-report-codex-ver/actions/runs/41277874562) on commit `8af498bbdd3225ee5fd0f03b34ae8bbe9c67b0f2` passed **446 Windows tests in 365.298 seconds**, compiled the single installer, verified the offline runtime report after install and reinstall, launched the installed app, preserved workspace data, completed uninstall, and uploaded artifact [13072646307](https://github.com/liorongoi/traffic-analysis-report-codex-ver/actions/runs/41277874562/artifacts/13072646307) (258,248,312 bytes; SHA-256 `9a52efeb9a87b9c3915d137ada2cc073a5146d9bb9558e941c293c1160a7480e`; expires 2026-11-08 05:51 UTC). Hosted CI does not substitute for a disconnected run on the target workstation or validate camera accuracy/performance.\n\nFoundation budget function uses physical/available RAM conservatively but does not allocate memory or prove throughput. Actual native probes, packaging and GPU operation remain planned.

## Optional offline YOLOX fine-tuning bundle

The core RoadLoom installer remains the single inference installer and does not include the large CUDA/PyTorch training stack. On a Windows staging machine, first prepare a relocatable Python 3.11/3.12 x64 runtime with CUDA-enabled PyTorch, ONNX and OpenCV, plus the YOLOX 0.3.0 source and its dependencies. Then create a separate transfer folder or ZIP:

```powershell
python scripts/stage-yolox-training.py --runtime E:\prepared\python --yolox-root E:\prepared\YOLOX-0.3.0 --output E:\transfer\RoadLoom-Training --zip
```

The stager performs no installation, network request, or authentication. It verifies the CUDA build and imports training/export modules, overlays the upstream `tools/` into an importable YOLOX package layout, records versions/device details and SHA-256 hashes, and includes the protected-split trainer plus the exact baseline checkpoint. It does not include reviewed image data. Use `--require-cuda` when the staging PC itself must have a working GPU; always run `verify-training.cmd` on the offline target to test its driver with a real CUDA tensor operation. Then use `finetune.cmd --dataset-zip ... --output ...` and import the resulting model pack through RoadLoom. The bundle may be large and remains separate from the app installer. This packages a preprepared environment; it does not prove training speed or CUDA compatibility on the target machine.
