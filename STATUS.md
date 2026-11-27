# 运行进度

2026-11-27，上海时间。A800 80GB 已完成两档原生 vLLM smoke。256 档实测 234–247 个视觉 token，1280 档实测 1107–1148 个视觉 token。每档 4/6 对输出逐 token 一致，分歧均出现在 COCO 长描述请求。

Pilot 从 14:47 开始，五档预算依次使用原生 AR 和 DFlash。正式全量与独立 GPU 诊断会在前序阶段通过后接续。

## 已归档的 pilot 预算

- 256 档共 360 对有效计时，E2E 加速 2.096×，token 一致 285/360。 [摘要](results/pilot-checkpoints/v256/summary.json) · [原始证据](results/pilot-checkpoints/v256/native-evidence.tar.gz)
- 512 档共 360 对有效计时，E2E 加速 2.075×，token 一致 288/360。 [摘要](results/pilot-checkpoints/v512/summary.json) · [原始证据](results/pilot-checkpoints/v512/native-evidence.tar.gz)
- 768 档共 360 对有效计时，E2E 加速 1.998×，token 一致 291/360。 [摘要](results/pilot-checkpoints/v768/summary.json) · [原始证据](results/pilot-checkpoints/v768/native-evidence.tar.gz)

## GPU 队列

- 2026-11-27T14:47:31+08:00 Starting pilot
