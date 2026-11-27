# 运行进度

2026-11-27，上海时间。A800 80GB 已完成两档原生 vLLM smoke。256 档实测 234–247 个视觉 token，1280 档实测 1107–1148 个视觉 token。每档 4/6 对输出逐 token 一致，分歧均出现在 COCO 长描述请求。

Pilot 从 14:47 开始，五档预算依次使用原生 AR 和 DFlash。正式全量与独立 GPU 诊断会在前序阶段通过后接续。

## 已归档的 pilot 预算

- 256 档通过 3 次重复，共 360 对请求。全请求 E2E 加速为 2.096×，输出 token 一致 285/360。跨预算分析将在五档结束后统一汇总。 [原始证据](results/pilot-checkpoints/v256/native-evidence.tar.gz) · [摘要](results/pilot-checkpoints/v256/summary.json)
