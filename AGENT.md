# AGENT.md — Vision Overhead 8×A100 执行手册

本手册只覆盖执行：**Model / Data / Environment / Launch / HF Upload**。目标是让外部 GPU 操作员不改任何代码，按顺序跑完 1 台 8×A100 80GB 节点上的 Conflict 1 实验（Multimodal Diffusion Speculative Decoding 的视觉开销）。

固定配置：`configs/vision_overhead_8gpu.yaml`（唯一事实来源），target `Qwen/Qwen3-VL-8B-Instruct`，TP=1，7 个 VelaDraft 并行 block drafter，vLLM 0.22.1。所有脚本都从该文件解析路径与 pin。

```bash
git clone https://github.com/LUOseu/visual-context-horizon-experiments.git
cd visual-context-horizon-experiments
```

## 1. Model

### 环境变量

| 变量 | 默认值 | 用途 |
| :--- | :--- | :--- |
| `MODEL_ROOT` | `./assets/models` | target 模型目录 |
| `CACHE_DIR` | `./assets/cache` | drafter checkpoint 与数据集下载缓存 |
| `HF_TOKEN` | 未设置 | gated 仓库下载；只从环境读取，不写入任何清单或日志 |
| `HF_ENDPOINT` | `https://huggingface.co` | HF 镜像地址（可选） |
| `JOBS` | `4` | 并行下载线程数 |

### Pin 一览

| 角色 | 仓库 | Revision | 状态 |
| :--- | :--- | :--- | :--- |
| Target | `Qwen/Qwen3-VL-8B-Instruct` | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | 必须 |
| Drafter | `khazic/veladraft-qwen3vl8b-200k-dflash` | `37a5eb63b12d851fe6114f3aeb87dbc7174dec3d` | 可运行 |
| Drafter | `khazic/veladraft-qwen3vl8b-200k-dflash2` | `def63d3a364adfc0ebac4af59955ddb392b4fa04` | 可运行 |
| Drafter | `khazic/veladraft-qwen3vl8b-200k-dspark` | `744db4cabb1c5d1b2f29f91a7da75882bc8f7820` | **blocked** |
| Drafter | `khazic/veladraft-qwen3vl8b-200k-domino` | `e85aad00bb83e1fbdcbf89e6f7c156e4ef944439` | **blocked** |
| Drafter | `khazic/veladraft-qwen3vl8b-200k-jetspec` | `0163faebb15cbcc73c77c07230d06f82b9166f44` | 可运行 |
| Drafter | `khazic/veladraft-qwen3vl8b-200k-dflare` | `cbddc83e0aa38aff652102f52d0bfa6d08cc0eee` | 可运行 |
| Drafter | `khazic/veladraft-qwen3vl8b-200k-dfly` | `24331eae8ac5dbe0f12390aa32c92f4e7efec5ab` | 可运行 |

`dspark` 与 `domino` 在 VelaDraft `e89526b` 上没有可用的 Qwen3-VL-8B native adapter（Domino 还缺 `DOMINO_FULL_PREFLIGHT.json` 与签名的 runtime-approval receipt）。两者仍可下载并校验，但调度器会跳过，`prepare`/`run` 会打印原因；这不是失败。**五个可运行 drafter 足以覆盖全部实验。**

### 下载

```bash
bash scripts/download_assets_8gpu.sh --models-only   # 并行下载 + 逐文件 SHA256 校验
bash scripts/download_assets_8gpu.sh --verify-only   # 无网络，只复核已有资产
```

target 解压到 `$MODEL_ROOT/Qwen3-VL-8B-Instruct`；drafter checkpoint 落在 `$CACHE_DIR/checkpoints/qwen3vl_8b/<method>/ckpt_final.pt`，供 `prepare_8gpu.sh` 导出。

## 2. Data

### 环境变量

| 变量 | 默认值 | 用途 |
| :--- | :--- | :--- |
| `DATA_ROOT` | `./assets/datasets` | textvqa/docvqa/chartqa/vqav2/coco_long_form 图像与元数据 |
| `CACHE_DIR` | `./assets/cache` | HR-Bench / MP-DocVQA / Video-MME 固定分片与视频 |

### 任务与切分

| 任务 | 轨道 | 预算 | 帧数 |
| :--- | :--- | :--- | :--- |
| `textvqa` `docvqa` `chartqa` `vqav2` `coco_long_form` `mathvista` | single image | 256–16384 | 1 |
| `hrbench` | single image（4K/8K 真实分辨率） | 256–4096 | 1 |
| `mp_docvqa` | multi frame | 2048–16384 | 8, 16 |
| `video_mme` | multi frame | 2048–16384 | 8, 16, 32, 64 |
| `ocrbench` `infovqa` | bonus single image | 512/2048/8192 | 1 |
| `text_control` | 零图像文本对照 | 256–16384 | — |

- 样本选择固定 `seed=42`；`workloads.per_task=100`，实验内每 cell 取 24（Exp4 取 32，wave0 取 2）。
- `mathvista`/`ocrbench`/`infovqa` 为可选数据：存在 `$DATA_ROOT/<task>/raw_manifest.json` 时自动纳入，否则在生成清单时打印 `task omitted` 并跳过。
- **不会放大图片，也不会合成像素。** 只有当样本原生像素能提供不少于预算的视觉 token 时才参与该 cell；达不到的 cell 记为 `pruned` 并打印原因。

```bash
bash scripts/download_assets_8gpu.sh --data-only
# 下载 HR-Bench / MP-DocVQA / Video-MME pin，并还原五个 QA 任务的图像树
```

数据下载依赖 `setup_8gpu.sh` 已组装 VelaDraft（QA 图像还原脚本位于 checkout 内）。

## 3. Environment

### 版本锁

| 组件 | 要求 |
| :--- | :--- |
| OS / GPU | Linux x86_64（glibc ≥ 2.35），1 节点 × 8 × A100 80GB |
| Python | `3.12.3` |
| PyTorch | `2.11.0+cu130`（CUDA 13.0） |
| Transformers | `5.11.0` |
| vLLM | `0.22.1`；`vllm/v1/sample/rejection_sampler.py` SHA256 必须是 `0e2be02449692e8f4aea22c4637727e8d1fc296fb413e5407b1aa79419bf7b41` |
| VelaDraft | commit `e89526b30b028d396005a3ff045238d829d6c24a` |
| Visual-KV hook | `matrix/visual_kv_hook.py` SHA256 在 `provenance/runtime_versions_8gpu.json` 中锁定，`prepare` 的 runtime 步骤会复核 |

### 组装与校验

```bash
export VELADRAFT_DIR=/data/veladraft     # 可选，默认 vendor/veladraft
bash scripts/setup_8gpu.sh               # 拉取 pin、拷贝方法目录、校验 adapter 与运行时
bash scripts/setup_8gpu.sh --dry-run     # 只打印计划
bash scripts/setup_8gpu.sh --assemble-only
bash scripts/setup_8gpu.sh --verify-only
```

运行时校验失败会直接退出并列出差异。仅调试时可 `VCH_SKIP_REJECTION_CHECK=1` 跳过 rejection sampler 校验；正式跑数不要使用。

### 准备训练好的 bytes（不重新训练）

```bash
bash scripts/prepare_8gpu.sh
```

依次完成：校验 target/数据 pin → 用 VelaDraft 的 exporter 导出五个可运行 drafter（训练字节不变）→ 生成 workload manifest（`$OUT_ROOT/manifests/workloads_manifest.json`）→ 锁定导出哈希 → 写 `runtime_manifest.json`。成功打印 `[PREPARE OK]`，失败打印 `[PREPARE BLOCKED]` 并退出。

常用参数：`--dry-run`、`--skip-exports`/`--lock-from-local`、`--allow-missing-data`、`--methods dflash,dflash2`。`--skip-missing-drafters` **只用于开发**；正式流程必须全量校验。

可选：先在 CPU 上跑计划/单元测试。

```bash
PYTHONPATH=methods/research/visual_context_horizon VCH_REPO_ROOT=$PWD \
  python3 -m unittest discover -s tests -v
```

torch 不可用时视觉 mask 用例会自动 skip；GPU 机上会执行。

## 4. Launch

### 先看计划

```bash
OUT_ROOT=/data/vch bash scripts/run_matrix_8gpu.sh --wave wave0_smoke --dry-run
OUT_ROOT=/data/vch bash scripts/run_matrix_8gpu.sh --wave wave1_main  --dry-run
OUT_ROOT=/data/vch bash scripts/run_matrix_8gpu.sh --wave wave2_bonus --dry-run
```

dry-run 会打印每个实验的 method/cell 数、每 lane 的 GPU、stage 数、唯一 cell 与请求估计、被剪枝的 cell、blocked drafter，以及 lane/stage 启动命令。无真实数据时自动使用 shape-only 合成清单；不会触碰 GPU。

### 三个 wave

| Wave | 内容 | GPU |
| :--- | :--- | :--- |
| `wave0_smoke` | HR-Bench 2 samples × 预算 256/16384 × `full`/`drop_all`，lossless 0\|0、CUDA graph、schema | 0–2 |
| `wave1_main` | Exp1 视觉预算扫描 ×4 lane，Exp2 drafter visual-KV 消融 ×2 lane，Exp3 并发 + rescue，Exp4 独立 profiling | 0–7（固定映射） |
| `wave2_bonus` | OCRBench + InfoVQA、APC ON/OFF、block width 4/8/16 | 0–7（LPT 均衡） |

`wave1_main` 的固定映射（与配置一致）：

| GPU | 实验 | 方法 |
| :--- | :--- | :--- |
| 0 | Exp1 | `ar`, `dflash` |
| 1 | Exp1 | `dflash2`（`dspark` 被跳过） |
| 2 | Exp1 | `jetspec`（`domino` 被跳过） |
| 3 | Exp1 | `dflare`, `dfly` |
| 4 | Exp2 | `ar`, `dflash`, `dflash2` |
| 5 | Exp2 | `dflare`（`dspark` 被跳过） |
| 6 | Exp3 + rescue | `ar` + 五个可运行 drafter |
| 7 | Exp4 profiling | `ar` + 五个可运行 drafter |

### 执行顺序

```bash
export OUT_ROOT=/data/vch
export MODEL_ROOT=/data/models
export DATA_ROOT=/data/datasets
export HF_TOKEN=...            # 如 target 为 gated

bash scripts/setup_8gpu.sh
bash scripts/download_assets_8gpu.sh
bash scripts/prepare_8gpu.sh

bash scripts/run_matrix_8gpu.sh --wave wave0_smoke
# 必须看到 [WAVE0 PASS]；否则先排查，不要继续
bash scripts/run_matrix_8gpu.sh --wave wave1_main
bash scripts/run_matrix_8gpu.sh --wave wave2_bonus
```

- 每个 wave 共 3 次重复；AR/SD 顺序按 repeat 交替；每 cell 前 3 个 warmup 不计时。
- 每个 stage 都是全新子进程，adapter bootstrap 在 `torch`/`vLLM` 导入前执行；AR 与 SD、不同 visual-KV 模式、不同 batch/宽度之间不共享 engine。
- Exp1–Exp3 不做 `torch.cuda.synchronize()`；六阶段 CUDA-event profiling 只在 Exp4 的独立进程中运行。
- 单个 cell OOM 会写 `oom=true` 的行，释放显存后继续同 lane 后续 cell。
- 运行目录：`$OUT_ROOT/runs/<wave>_<UTC>/`，含 `plan.json`、`config_resolved.json`、`workload_manifest.json`、`lanes/`、`logs/`、`telemetry/`、`cells/`、`summary.json|csv`；wave0 的判定写到 `summary.json.wave0`。

### 进度、续跑与失败重试

```bash
bash scripts/status_8gpu.sh --wave wave1_main
bash scripts/status_8gpu.sh --run-dir $OUT_ROOT/runs/wave1_main_20261127T101500Z --json

# 中断后用同一 run-dir 续跑，只补缺失的 (repeat, sample)
bash scripts/run_matrix_8gpu.sh --wave wave1_main \
  --run-dir $OUT_ROOT/runs/wave1_main_20261127T101500Z --resume

# 重试已标记 failed/oom 的 cell
bash scripts/run_matrix_8gpu.sh --wave wave1_main \
  --run-dir $OUT_ROOT/runs/wave1_main_20261127T101500Z --resume --retry-failed
```

## 5. HF Upload

每个 wave 跑完后打包上传（默认 HF Dataset）：

```bash
bash scripts/pack_and_upload.sh wave0_smoke
bash scripts/pack_and_upload.sh wave1_main
bash scripts/pack_and_upload.sh wave2_bonus
```

| 变量 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `HF_TOKEN` | 未设置 | 写入权限；只从环境读取 |
| `HF_REPO_PRIVATE` | `0` | `1` 时创建私有 Dataset |
| `HF_RESULT_REPO` | `khazic/visual-context-horizon-results` | 目标 Dataset 仓库，不存在会自动创建 |

上传路径为 `<repo>/<wave>/`。上传失败时自动生成干净本地包 `vch_<wave>_<gpu>_<date>.tar.gz`；`--no-upload` 强制只生成本地包，`--run-dir DIR` 指定要打包的运行目录。

包内包含：`summary.json`、`summary.csv`、`summary_matrix.json/csv`、`plan.json`、`runtime_manifest.json`、`config_resolved.json`、`workload_manifest.json`、`lanes/`、`logs/`、`telemetry/`、`per_request_telemetry.tar.gz`（逐请求 JSONL + cell 状态）、`STATUS.md`/`SHA256SUMS`。

**始终排除**：模型权重（`.pt`/`.safetensors`/`.bin`/`.ckpt`）、原始图像/视频、parquet、profiler trace/profile、`token.txt`/`.env`/私钥等凭据。任何脚本都不会把 `HF_TOKEN` 写进清单、日志或结果包。
