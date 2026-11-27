# AGENT.md — Visual Context Horizon 实验启动手册（GPU 机执行版）

本手册面向有 GPU 的实验执行同学。所有前置组装、参数锁定、结果打包逻辑都已写成仓库里的脚本，**不需要改代码、不需要手动打补丁、不需要开 PR**，按顺序配置路径并运行脚本即可。

## 1. 获取代码与一键组装

实验归档仓库为 [LUOseu/visual-context-horizon-experiments](https://github.com/LUOseu/visual-context-horizon-experiments)；评测框架来自 [LUOseu/VelaDraft](https://github.com/LUOseu/VelaDraft)，锁定 commit `e89526b30b028d396005a3ff045238d829d6c24a`，由脚本自动浅拉取组装。

```bash
git clone https://github.com/LUOseu/visual-context-horizon-experiments.git
cd visual-context-horizon-experiments

# 拉取 VelaDraft 锁定 commit、拷入 methods/research/visual_context_horizon、
# 应用 methods/native-eval.patch，并严格校验 Python / torch / transformers / vLLM
bash scripts/setup.sh
```

默认组装置于 `vendor/veladraft`，可用 `VELADRAFT_DIR` 改路径。其他模式：

```bash
bash scripts/setup.sh --dry-run        # 只打印计划，不写文件
bash scripts/setup.sh --assemble-only  # 只组装，跳过版本校验
bash scripts/setup.sh --verify-only    # 复核已有组装与运行时
```

## 2. 运行环境要求

`setup.sh` 启动前强校验以下版本，并核对 `provenance/runtime_versions.json` 与 `provenance/runtime_manifest.json` 记录的核心文件 SHA256：

| 组件 | 要求版本 / 规格 | 说明 |
| :--- | :--- | :--- |
| **GPU** | 单卡 80GB 显存（基准机为 A800 80GB PCIe） | `TP=1`，BF16，batch size = 1 |
| **Python** | `3.12.3` | Linux x86_64（glibc >= 2.35） |
| **PyTorch** | `2.11.0+cu130`（CUDA `13.0`） | 强校验 |
| **Transformers** | `5.11.0` | 强校验 |
| **vLLM** | `0.22.1` | 强校验 `vllm/v1/sample/rejection_sampler.py` SHA256：`0e2be02449692e8f4aea22c4637727e8d1fc296fb413e5407b1aa79419bf7b41` |

如果 vLLM 校验失败，按提示重装：

```bash
pip install --no-deps --force-reinstall vllm==0.22.1
```

## 3. 模型与数据集准备

### 模型与 Checkpoint（见 [dflash.json](methods/research/visual_context_horizon/configs/dflash.json)）

| 角色 | HuggingFace 仓库 / 文件 | 锁定 Revision | 校验说明 |
| :--- | :--- | :--- | :--- |
| **Target 模型** | `Qwen/Qwen3-VL-8B-Instruct` | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | 自动校验 [target_files.json](methods/research/visual_context_horizon/configs/target_files.json) 内 14 个文件 SHA256 |
| **Drafter 权重** | `khazic/veladraft-qwen3vl8b-200k-dflash` 的 `ckpt/ckpt_final.pt`（75K step） | `37a5eb63b12d851fe6114f3aeb87dbc7174dec3d` | 下载后存为 `dflash_final.pt`，SHA256 `86b40b550dae022d8f8669302022c50aaf36441c6f36c1f0adde6628f5c6135c` |

### 5 类实验数据集（样本清单见 [workloads/selected-500.json](workloads/selected-500.json)，`sample_seed=42`）

| 任务名 | 数据来源 | 默认子目录 |
| :--- | :--- | :--- |
| `textvqa` | 归档 256 题 TextVQA，图像从固定 Parquet 分片还原 | `$DATA_ROOT/textvqa` |
| `docvqa` | 归档 256 题 DocVQA，图像从固定 Parquet 分片还原 | `$DATA_ROOT/docvqa` |
| `chartqa` | 归档 256 题 ChartQA，图像从固定 Parquet 分片还原 | `$DATA_ROOT/chartqa` |
| `general_vqa` | VQAv2 validation（每张 COCO 图 1 问 10 答） | `$DATA_ROOT/general_vqa` |
| `long_form` | COCO `val2017` 图像 + 固定长描述 prompt | `$DATA_ROOT/long_form` |

数据脚本只下载 100 张 VQAv2 图与 100 张 COCO `val2017` 图（来自官方 JPEG 地址），无需 6GB 级完整归档；TextVQA/DocVQA/ChartQA 的 256 行题目元数据已随仓库提供在 `data/qa/`，图像字节从锁定 revision 的 Parquet 分片按 SHA256 还原。

### 一键下载 / 校验

默认路径为仓库内 `assets/`，所有路径均可用环境变量覆盖：

```bash
export TARGET=/path/to/Qwen3-VL-8B-Instruct   # 默认 assets/Qwen3-VL-8B-Instruct
export CHECKPOINT=/path/to/dflash_final.pt    # 默认 assets/dflash_final.pt
export DATA_ROOT=/path/to/datasets            # 默认 assets/datasets
export GPU_ID=0

# 下载模型 + 数据；断点续传，每个文件校验固定 SHA256
bash scripts/download_assets.sh

# 只下载模型 / 只下载数据 / 只校验已有资产
bash scripts/download_assets.sh --models-only
bash scripts/download_assets.sh --data-only
bash scripts/download_assets.sh --verify-only
```

下载缓存位于 `assets/cache`（可用 `CACHE_DIR` 覆盖），可在校验通过后删除。`--data-only` 需要先完成 `setup.sh`，因为图像还原会导入 VelaDraft 的 `research.visual_context_horizon`。

### 预处理（导出 drafter + 冻结清单）

```bash
bash scripts/prepare.sh
```

脚本依次完成：

1. 校验 `TARGET` 包含全部 14 个固定文件、`CHECKPOINT` 存在、`DATA_ROOT` 五类数据齐全。
2. 用 `export_checkpoint.py` 把 checkpoint 导出到 `artifacts/dflash_export`。
3. 用 `prepare_runtime.py` 写 `artifacts/runtime_manifest.json`。
4. 用 `prepare_workloads.py` 写 `workloads/manifest.json`，并校验图像 SHA256 与 `workloads/selected-500.json` 的前 100 条规范顺序。

已有完整产物会被验证后复用，不会覆盖；`--dry-run` 只打印计划。路径分别可用 `DRAFT_DIR`、`RUNTIME_MANIFEST`、`WORKLOAD_MANIFEST` 覆盖。

## 4. 实验配置与一键启动

### 锁定参数（已写入脚本，无需手动修改）

- `method=dflash`、`TP=1`、`batch_size=1`、BF16 确定性 greedy、自然 EOS。
- `speculative_tokens=7`（B8）、`max_new_tokens=512`、`max_model_len=8192`、`sd_cudagraph_mode=PIECEWISE`、`repeats=3`。
- `budgets=[256, 512, 768, 1024, 1280]`、`horizons=[32, 64, 128, 256, 512]`、`sample_seed=42`。
- Pilot 每任务 24 样本（360 个计时配对）；Full 每任务 100 样本（共 500 样本 × 5 预算 × 3 重复）。

### 按当前进度依次启动（进度见 [STATUS.md](STATUS.md)）

当前归档已包含 `smoke` 及 pilot 的 `v256 / v512 / v768 / v1024`；新机器从 smoke 开始，然后补齐剩余阶段：

```bash
# 1. 新机器必跑：冒烟验证（2 样本 × 预算 256/1280 × 1 次重复，约几分钟）
bash scripts/run_stage.sh smoke

# 2. 补齐 Pilot 最后一档 v1280（24 样本 × 3 重复 = 360 配对）
bash scripts/run_stage.sh pilot_v1280

# 3. Pilot 规模 GPU Diagnostics（五档预算，采集阶段耗时与验证计数）
bash scripts/run_stage.sh diagnostics

# 4. 全量 Full（100 样本 × 5 预算 × 3 重复）并自动生成分析与图表
bash scripts/run_stage.sh full
```

`GPU_ID` 必须是单个整数（`TP=1`），例如 `GPU_ID=1 bash scripts/run_stage.sh full`。每次运行的产物在 `runs/<stage>-<UTC 时间戳>/`，日志在 `logs/`，最后会写 `RUN_META.json` 并记录 `runs/<stage>.latest`。

## 5. 实验结果传回方式

**无需手动整理文件、修改 `STATUS.md` 或提 PR。** 每跑完一个阶段，直接执行：

```bash
# <stage> 对应 pilot_v1280 / diagnostics / full
bash scripts/pack_and_upload.sh <stage>
```

脚本会自动：

- 校验该阶段产物完整（v1280 必须为 360 配对 / 3 重复且资源校验通过；diagnostics 必须有诊断汇总；full 必须有报告、汇总、盈亏点与图表）。
- 以 `results/pilot-checkpoints/v256/` 的格式生成 `summary.json` 与 `native-evidence.tar.gz`，自动过滤模型权重（`.pt` / `.safetensors` / `.bin` / `.ckpt`）、原始图片、profiler 轨迹与缓存。
- 记录本机 GPU 型号与 UUID，刷新 `STATUS.md` 与 `SHA256SUMS`，并本地提交。
- 若本机已配置 GitHub 推送权限，自动推送到专属结果分支 `results/<stage>-<gpu-slug>`（例如 `results/full-nvidia-a800-80gb-pcie`）；若无推送权限，则在仓库根目录生成 `results-<stage>-<gpu-slug>-<时间戳>.tar.gz` 干净压缩包，直接发回即可，解包说明见包内 `README.txt`。

可选参数：`--run-dir DIR` 指定运行目录（默认读取 `runs/<stage>.latest`），`--no-push` 强制只生成本地压缩包。

## 6. 需要人工提供的本地文件

**无。** Target 模型、drafter checkpoint、五个数据集均可从公开地址下载并逐字节校验。仓库内的 `data/qa/{textvqa,docvqa,chartqa}/` 是小型题目元数据，已随 git 提供。唯一的可选本地资产是自备的完整数据集镜像；只要其 `$DATA_ROOT/<task>/` 下图像与元数据 SHA256 与清单一致，`prepare.sh` 会直接复用，无需重新下载。
