# 07 — Hardware profiles and offline deployment

Target: Windows workstation, 16/48 GB RAM, RTX A4000 or other graphics, one active user. Development execution on this Mac does not establish Windows or GPU compatibility.

An on-demand local host report now reads CPU model/logical cores, OS-reported physical and available memory, GPU/driver details when an inventory utility is present, ONNX Runtime provider names, and free disk capacity. It does not load a vision model. Provider names are marked advertised only; GPU execution remains unverified until the selected model successfully runs on that provider. Storage throughput is not measured. The report proposes a conservative CPU memory profile from current capacity. The Windows implementation and target measurements still require native workstation validation. RAM profile and GPU choice are independent. Integrate bounded decoding/tiling/frame queues; avoid eager model loading and simultaneous inference/training memory contention. Proposed budgets reserve OS/application headroom and must be measured on both target classes; they are not performance guarantees.

## Packaging design

Create immutable, versioned prerequisite/runtime/model packs plus small application updates. Stage pinned wheels, native libraries, local models, codecs, timezone data/boundaries, fonts and scripts on an internet-connected build machine. Record software/weights/data licenses separately and create a content manifest. Include all actual required assets in copy-ready media. No package managers/model pulls/CDNs/telemetry during installed operation or installation.

Installation and normal operation must not need administrator privileges. Use %LOCALAPPDATA%, current-user shortcuts and private runtimes. Local folders and authorized UNC shares are required. Autostart and credential handling remain design/testing work. Do not install a system service/global runtime. Bind a development API to loopback only; authenticated LAN deployment, TLS/certificate handling and host firewall instructions must be explicitly tested. Missing packs/unsupported GPU must fail clearly or use an explicitly selected CPU mode, never silently download/substitute a model.

Installation performs resumable content/native checks; normal launch reuses a receipt bound to exact content. App updates journal changes and preserve mission DB/evidence/checkpoints. Validate actual EXE contents and copied-file hashes independently, then verify native Windows offline installation/update/rollback on target hardware.

Foundation budget function uses physical/available RAM conservatively but does not allocate memory or prove throughput. Actual native probes, packaging and GPU operation remain planned.
