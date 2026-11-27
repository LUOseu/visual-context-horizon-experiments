# Visual Input Budget × Generation Horizon

截至 2026-11-27 的运行记录。样本清单和启动协议见 [README](README.md)。

## 已验证的前置条件

- 冻结数据包含 VQAv2 validation、TextVQA validation、DocVQA validation、ChartQA test 和 COCO val2017。GPU 机器已按图像 SHA256 构建每组 100 样本的正式清单。三项仓库 QA 任务复用历史 256 题原始数据和人工标注。
- 远端为 NVIDIA A800 80GB PCIe，CUDA 驱动 580.105.08。固定环境为 vLLM 0.22.1、PyTorch 2.11.0+cu130、Transformers 5.11.0。
- Qwen3-VL-8B target 的四个权重分片均通过 SHA256 校验。历史 B8 DFlash checkpoint 导出的 36 个张量逐字节保留。实验准入按原生 standard greedy verifier 加空间 MRoPE 适配器检查，AR/SD token 一致性作为观测指标。
- 本机用固定 target 元数据和 Transformers 5.12.1 做了处理器预检。一张 DocVQA 原图的五档实测网格对应 252、475、744、1008、1240 个视觉 token。正式运行记录固定 5.11.0 版本下 vLLM 实际生成的网格。
- 第一次 GPU smoke 在 AR 启动时发现 ninja 未进入 PATH。保留失败日志，在第二次运行中将固定环境的可执行目录加入 PATH。

## GPU smoke

两档都通过同一预算的 AR/SD 输入比对、实际视觉网格检查、native standard greedy verifier 检查和提交轨迹校验。每档 6 对请求，覆盖 ChartQA、TextVQA 和 COCO 长描述，每个任务各 2 条。

| 名义视觉预算 | 实测视觉 token 范围 | E2E 加速 | 输出 token 完全一致 |
| ---: | ---: | ---: | ---: |
| 256 | 234–247 | 2.081× | 4/6 |
| 1280 | 1107–1148 | 2.077× | 4/6 |

两档各两次输出序列分歧均来自长描述请求。两档共同样本有 2 条达到 256 个输出 token，512 个输出 token 的共同样本数为 0。加速数字来自单次 smoke，正式比较使用三个配对重复。

## 正式阶段

Pilot 于 14:47（上海时间）启动，纳入五类任务各 24 条样本和全部五档视觉预算。Pilot 通过后，队列继续每类 100 条的全量实测和独立诊断。正式 E2E、decode-only、break-even 和任务质量等待计时结束后按原始记录汇总。
