# Visual Input Budget × Generation Horizon

VelaDraft 原生 vLLM 实验档案。比较 Qwen3-VL-8B-Instruct 的 AR 和空间 DFlash B8 在不同视觉预算、自然生成长度下的延迟与任务质量。

在 GPU 机上执行实验请直接按 [AGENT.md](AGENT.md) 操作：`setup.sh` → `download_assets.sh` → `prepare.sh` → `run_stage.sh <stage>` → `pack_and_upload.sh <stage>`。

| 阶段 | 状态 | 入口 |
| --- | --- | --- |
| GPU smoke | 256 和 1280 两档通过，各 6 对请求 | [报告](results/smoke/REPORT.md) · [汇总](results/smoke/summary.csv) · [原始观测](results/smoke/native/) |
| Pilot | 运行中。五类任务各 24 个样本，五档预算，三次配对重复 | [进度](STATUS.md) |
| 全量 | 排在 pilot 之后。五类任务各 100 个样本 | [冻结样本清单](workloads/selected-500.json) |
| GPU 诊断 | 排在全量之后。包含独立阶段计时与验证计数 | [协议](methods/research/visual_context_horizon/README.md) |

Native standard greedy verifier 保留 target 的提交规则。实测 AR/SD token 一致率独立记录。正式效果由三个计时重复和样本级配对区间评估。Smoke 的数字用于验证运行链路。

复现所需的 VelaDraft 基础提交和来源见 [源码记录](provenance/source.json)。将方法目录中的研究脚本放在该提交的同名目录，并应用 [native evaluator 补丁](methods/native-eval.patch)。[运行环境](provenance/runtime_versions.json) 和 [DFlash 准入清单](provenance/runtime_manifest.json) 记录版本及检查点身份。样本清单包含数据集、分割、问题、真实标注与图像 SHA256。原始图像可按实验协议从各固定来源复原。
