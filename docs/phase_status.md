# Phase Status

本文档记录阶段执行状态。`DEVELOPMENT_PLAN.md` 是需求和验收标准的唯一来源。

| Phase | 内容 | 自动验收 | 人工验收 | 状态 |
|---|---|---:|---:|---|
| Phase 0 | 仓库基线与开发工具 | 通过 | 待验收 | AUTO_PASS |
| Phase 1 | 上游审计与模型契约 | 通过 | 待验收 | AUTO_PASS |
| Phase 2 | 核心类型、配置与 FakeBackend | 通过 | 待验收 | AUTO_PASS |
| Phase 3 | 谱图读取与预处理 | 通过 | 待验收 | AUTO_PASS |
| Phase 4 | 真实 CPICANN 模型加载与推理 | 通过 | 待验收 | AUTO_PASS |
| Phase 5 | 类别目录、元素过滤与置信度重算 | 未开始 | 未开始 | TODO |
| Phase 6 | 批量任务、输出、诊断和中文报告 | 未开始 | 未开始 | TODO |
| Phase 7 | CLI 产品化 | 未开始 | 未开始 | TODO |
| Phase 8 | Streamlit Web 应用 | 未开始 | 未开始 | TODO |
| Phase 9 | FastAPI 接口 | 未开始 | 未开始 | TODO |
| Phase 10 | Docker、CI/CD 与供应链控制 | 未开始 | 未开始 | TODO |
| Phase 11 | 真实模型 Golden Test、发布候选与验收 | 未开始 | 未开始 | TODO |

## Phase 0 Notes

- 已创建项目骨架、工具配置、最小 CLI 和基础测试。
- 不包含真实模型、上游审计、类别目录或推理逻辑。
- 自动验收命令已于 2026-07-12 通过。
- 本阶段仍需人工验收后再进入 Phase 1。

## Phase 1 Notes

- 已审计本地 GitHub fork、WPEM/CPICANN、HF source mirror、HF pretrained repository 和 `WPEMPhase==0.1.1` wheel。
- 已创建模型合同、预处理合同和模型接入 ADR。
- 已创建只读检查脚本 `scripts/inspect_upstream.py`。
- 未下载真实权重；checkpoint 实际顶层键、state_dict 前缀和权重再分发许可仍为 UNKNOWN。
- 未找到 `0-norm.txt`、`1-norm.txt`、`3-norm.txt`；这些 fixture 的 golden 统计需在 Phase 3 补齐。
- 自动验收命令已于 2026-07-12 通过。
- 本阶段仍需人工验收后再进入 Phase 2。

## Phase 2 Notes

- 已定义核心 Pydantic schema、稳定错误码、配置加载、模型 manifest/registry、InferenceBackend 协议和确定性 FakeBackend。
- 已创建最小 `PredictionService`，可用 FakeBackend 驱动单样品预测结果。
- 配置加载顺序已固定为 CLI 参数 > 环境变量 > 配置文件 > 默认值。
- 未实现真实 CPICANNBackend、真实 catalog、元素过滤、置信度重算、Web 或 API。
- 自动验收命令已于 2026-07-12 通过。
- 本阶段仍需人工验收后再进入 Phase 3。

## Phase 3 Notes

- 已实现 `.txt`、`.csv`、`.xy` 谱图读取，扩展名大小写不敏感。
- 已支持空格、Tab、逗号分隔，两列和三列输入；三列按 legacy 合同使用第二列减第三列作为强度。
- 已记录/处理表头、非法行、倒序角度、重复角度、NaN/Inf、空文件和不支持扩展名。
- 已实现 `legacy-cpicann-v1` 固定 10-80 度、4500 点、最大强度归一到 100 的预处理输出。
- 已加入可再分发的合成 `0-norm.txt`、`1-norm.txt`、`3-norm.txt` fixtures 和 golden hash；它们不是上游真实样品。
- 未实现真实模型、Web、API、catalog 或元素过滤。
- 自动验收命令已于 2026-07-12 通过。
- 本阶段仍需人工验收后再进入 Phase 4。

## Phase 4 Notes

- 已实现 CPICANN 单相网络最小推理定义、CPICANNBackend、manifest YAML、权重 SHA-256 校验和模型目录解析。
- 已实现 `doctor` 服务和 `cpicann-xrd doctor --backend fake` 命令。
- 已实现 `scripts/verify_model_bundle.py`、`scripts/convert_checkpoint.py` 和显式 opt-in 的 `scripts/download_model.py`。
- 已将本地真实 checkpoint 放入 `models/cpicann-single-d1/CPICANNsingle_phase_D1.pth`，原始 SHA-256 为 `d2e898bb4b7482cd7b14953feac437b053617815746f025a6b88ca014e51be98`。
- 原始 checkpoint 含 optimizer 对象，已按受控流程转换为纯 `state_dict` 运行期权重 `CPICANNsingle_phase_D1.state_dict.pth`，SHA-256 为 `0f3a452da5218df46eaa37f7d0dadb388e08cdeafa6e24e9d6e2e93512bb2e9a`。
- 无权重时 `doctor --backend cpicann` 和 verify 脚本返回 `MODEL_NOT_INSTALLED`，不输出敏感 token 或栈追踪。
- 真实模型测试已标记为 `model`，未配置 `CPICANN_MODEL_DIR` 时跳过；配置本地真实权重后 smoke test 已通过。
- 已使用 `samples/CPICANN识别/0-norm.txt`、`1-norm.txt`、`3-norm.txt` 做样品级真实 smoke，覆盖谱图读取、预处理、真实模型前向和 top-5 class index 稳定性。
- 未提交真实权重；未实现 catalog、元素过滤、Web、API 或带物相名称的真实预测 golden test。
- 自动验收命令已于 2026-07-12 通过。
- 本阶段仍需人工验收后再进入 Phase 5。
