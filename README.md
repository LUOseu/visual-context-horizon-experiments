# Visual Input Budget × Generation Horizon

VelaDraft 原生 vLLM 实验档案。比较 Qwen3-VL-8B-Instruct 的 AR 和空间 DFlash B8 在不同视觉预算、自然生成长度下的延迟与任务质量。

在 GPU 机上执行实验请直接按 [AGENT.md](AGENT.md) 操作：`setup_8gpu.sh` → `download_assets_8gpu.sh` → `prepare_8gpu.sh` → `run_matrix_8gpu.sh --wave <wave0_smoke|wave1_main|wave2_bonus>` → `pack_and_upload.sh <wave>`。下表的 smoke/pilot/full 是已归档的旧单卡实验。

| 阶段 | 状态 | 入口 |
| --- | --- | --- |
| GPU smoke | 256 和 1280 两档通过，各 6 对请求 | [报告](results/smoke/REPORT.md) · [汇总](results/smoke/summary.csv) · [原始观测](results/smoke/native/) |
| Pilot | 运行中。五类任务各 24 个样本，五档预算，三次配对重复 | [进度](STATUS.md) |
| 全量 | 排在 pilot 之后。五类任务各 100 个样本 | [冻结样本清单](workloads/selected-500.json) |
| GPU 诊断 | 排在全量之后。包含独立阶段计时与验证计数 | [协议](methods/research/visual_context_horizon/README.md) |

Native standard greedy verifier 保留 target 的提交规则。实测 AR/SD token 一致率独立记录。正式效果由三个计时重复和样本级配对区间评估。Smoke 的数字用于验证运行链路。

8-GPU 套件的 VelaDraft 基线提交、vLLM rejection sampler 与 drafter visual-KV hook 的 SHA256 见 [运行环境](provenance/runtime_versions_8gpu.json)；`setup_8gpu.sh` 会浅拉取该提交并把方法目录拷入 checkout，无需手工打补丁。[样本清单](workloads/selected-500.json) 包含数据集、分割、问题、真实标注与图像 SHA256；原始图像由 `download_assets_8gpu.sh` 从固定来源复原。旧单卡实验的来源记录仍在 [source.json](provenance/source.json) 与 [native-eval.patch](methods/native-eval.patch)。
