# CPICANN-XRD-App × XDecomposer 集成开发与验收计划

> 文档用途：供 Codex 按阶段实施 XDecomposer 集成；供项目负责人、科研人员和运维人员逐阶段验收。<br>
> 基线日期：2026-07-14<br>
> 依据文档：`xdecomposer_feasibility.md`<br>
> 目标仓库：`madokaiskami/CPICANN-XRD-App`<br>
> 建议发布版本：`v0.2.0`<br>
> 默认稳定能力：CPICANN 单相物相候选排序<br>
> 新增可选能力：XDecomposer 多相谱图分解，以及 XDecomposer → CPICANN 的分相识别工作流

---

## 1. 集成结论

### 1.1 技术结论

**可以集成。**

但第一版不得采用“把 XDecomposer 直接安装进现有 CPICANN Python 环境”的方式。推荐架构是：

```text
Browser / CLI / API Client
            │
            ▼
     CPICANN-XRD-App API
            │
      ┌─────┴──────────────┐
      ▼                    ▼
CPICANN capability    XDecomposer capability
single-phase rank     multiphase decomposition
      │                    │
      └──────────┬─────────┘
                 ▼
      Orchestration / Report
```

部署层面采用独立容器：

```text
docker compose
├── web
├── api
├── cpicann-worker
└── xdecomposer-worker
```

### 1.2 为什么必须隔离

根据可行性文档，当前两套运行环境存在明显差异：

| 项目 | CPICANN-XRD-App | XDecomposer 上游 |
|---|---|---|
| Python | `>=3.11,<3.14` | 3.10 |
| PyTorch | CPU 源，`>=2.5,<3` | 2.10.0 |
| CUDA | 默认 CPU | CUDA 12.8 依赖 |
| 额外依赖 | 轻量推理栈 | TensorFlow、PyG、Triton 等 |
| 输入长度 | 4500 | 3500 |
| 归一化 | CPICANN 兼容协议 | 常见路径为 max normalization |
| 输出语义 | 固定类别 logits | 分解谱图 + activity logits |

同进程安装会带来 Python、PyTorch、CUDA、TensorFlow和科学计算包的依赖冲突，也会扩大 checkpoint 加载失败对现有 CPICANN 服务的影响范围。

### 1.3 产品边界

集成完成后，产品应有三个明确模式：

```text
single-phase
    CPICANN 对原始谱图做单相候选排序

multiphase-decomposition
    XDecomposer 分解混合谱，不调用 CPICANN

multiphase-identification
    XDecomposer 分解 → 每个活跃 component 交给 CPICANN 排名
```

不得把三种模式的分数混为一谈：

- `cpicann_filtered_confidence`：元素过滤后候选集合中的条件分类分数；
- `xdecomposer_activity_probability`：某个输出 slot 是否活跃；
- `xdecomposer_estimated_weight`：基于分解谱强度估算的贡献值；
- `reference_similarity`：XDecomposer 参考库匹配相似度。

`estimated_weight` 在完成科学验证之前，不得称为“定量相含量”。

---

## 2. 集成的前置闸门

只有以下条件全部满足，才能进入真实权重集成：

1. XDecomposer 源码使用许可已确认并保留 MIT notice；
2. separator checkpoint 的来源、许可、文件名、大小和 SHA-256 已确认；
3. MAE 预训练 checkpoint 的来源、许可、文件名、大小和 SHA-256 已确认；
4. reference bank 或数据库的来源及再分发许可已确认；
5. MP20、RRUFF 或其他数据衍生物的使用边界已确认；
6. 真实权重只能通过本地挂载或批准的私有存储进入运行环境；
7. Git 仓库、Docker 镜像和公开 Release 中不得包含未获授权的权重或数据库；
8. checkpoint 必须先做 SHA-256 校验，再由隔离 worker 加载；
9. 未通过上述闸门时，只能使用 stub/fake backend 完成产品层开发。

任何一项不满足，都不得对外宣称“XDecomposer 已集成可用”。

---

## 3. 最终工作目标

Codex 的最终工作目标不是简单复制上游代码，而是完成一个稳定、可追溯、可选启用的多相分析能力：

```text
原始 XRD
   │
   ▼
输入验证与物理坐标解析
   │
   ▼
XDecomposer 专用预处理
10–80° / 3500 点 / 明确归一化版本
   │
   ▼
XDecomposer 分解
component patterns + activity logits
   │
   ├── 重建谱和残差分析
   ├── activity threshold
   └── component 稳定排序
   │
   ▼
每个活跃 component 映射回物理 2θ 轴
   │
   ▼
CPICANN 专用预处理
10–80° / 4500 点 / legacy-cpicann-v1
   │
   ▼
CPICANN 分类
   │
   ▼
元素约束 + 过滤后 softmax + Top-K
   │
   ▼
分相识别报告、CSV、图像和元数据
```

最终用户应能：

1. 上传一个混合 XRD 文件；
2. 选择“多相分解”或“多相分解并识别”；
3. 配置最大 component 数和 activity threshold；
4. 配置 CPICANN 元素约束；
5. 获得原谱、重建谱、残差图；
6. 查看每个活跃 component 的分解谱；
7. 查看每个 component 的 CPICANN Top-K；
8. 下载 CSV、JSON、PNG、Markdown 和 ZIP；
9. 在元数据中追溯代码、模型、权重、配置、输入和设备。

---

## 4. 不可变工程规则

Codex 在所有阶段都必须遵守：

### 4.1 不破坏当前能力

- CPICANN 必须继续作为默认能力；
- XDecomposer 未启动、不可用或资产缺失时，单相模式仍应正常工作；
- 不得修改现有 CPICANN 4500 点预处理语义；
- 不得把 XDecomposer 3500 点数组直接 resize 后送入 CPICANN；
- 所有网格转换必须通过物理 2θ 坐标插值。

### 4.2 接口分离

不得把 XDecomposer 实现为现有 `InferenceBackend.predict_logits()` 的另一个实现。

推荐协议：

```python
class PhaseRankingBackend(Protocol):
    def predict_logits(self, x: Tensor) -> Tensor: ...


class PatternDecompositionBackend(Protocol):
    def decompose(self, request: DecompositionRequest) -> DecompositionResult: ...
```

### 4.3 资产安全

- 不自动下载真实权重；
- 不提交 `.pt`、`.pth`、reference bank 或数据集；
- 不信任用户上传的 checkpoint；
- 资产校验失败必须 fail closed；
- 任何 `torch.load` 仅限已授权且哈希匹配的本地资产；
- 真实 token、云端密钥和数据库凭据只能放入 secret/environment。

### 4.4 科学表述

报告必须明确：

- component 是模型分解结果，不等同于已确认物相；
- activity probability 不等同于分类置信度；
- estimated weight 不等同于 Rietveld 定量相含量；
- CPICANN Top-K 是固定类别库中的候选排序；
- 训练库之外的新相可能无法识别；
- 多相结果仍需实验专家、峰匹配或精修复核。

### 4.5 每阶段交付纪律

每个阶段都必须：

1. 使用独立分支；
2. 先输出只读计划；
3. 只修改该阶段允许的范围；
4. 添加测试；
5. 运行验收命令；
6. 输出修改文件列表；
7. 输出未解决问题；
8. 不自动提交和推送，除非人明确授权；
9. 不声称未运行的测试已经通过。

---

## 5. 建议目录结构

```text
CPICANN-XRD-App/
├── configs/
│   └── models/
│       ├── xdecomposer.example.yaml
│       └── xdecomposer-stub.yaml
│
├── docker/
│   ├── Dockerfile.xdecomposer
│   └── xdecomposer-entrypoint.sh
│
├── models/
│   └── xdecomposer/
│       ├── README.md
│       └── manifest.example.yaml
│
├── services/
│   └── xdecomposer_service/
│       ├── pyproject.toml
│       ├── lockfile
│       ├── src/
│       │   └── xdecomposer_service/
│       │       ├── main.py
│       │       ├── settings.py
│       │       ├── schemas.py
│       │       ├── assets.py
│       │       ├── preprocessing.py
│       │       ├── backend.py
│       │       ├── upstream_adapter.py
│       │       ├── artifacts.py
│       │       └── health.py
│       └── tests/
│
├── src/cpicann_xrd/
│   ├── decomposition/
│   │   ├── __init__.py
│   │   ├── schemas.py
│   │   ├── client.py
│   │   ├── service.py
│   │   ├── orchestration.py
│   │   └── exceptions.py
│   ├── reports/
│   │   └── decomposition_report.py
│   ├── api/
│   │   └── decomposition_routes.py
│   └── web/
│       └── decomposition_page.py
│
├── tests/
│   ├── decomposition/
│   ├── integration/
│   └── scientific/
│
├── compose.xdecomposer.yaml
└── docs/
    ├── xdecomposer_feasibility.md
    ├── xdecomposer_integration_plan.md
    ├── xdecomposer_asset_licenses.md
    ├── xdecomposer_runtime.md
    ├── xdecomposer_scientific_validation.md
    └── xdecomposer_operations.md
```

---

# 6. 分阶段实施计划

---

## Phase XD-0：冻结 CPICANN 基线

### 目标

在引入 XDecomposer 前，确认当前 CPICANN 单相版本有一个可回滚、可重复验证的稳定基线。

### Codex 工作

1. 只读检查当前分支、tag、工作区和远程仓库；
2. 确认默认分支为 `main`；
3. 确认 CPICANN 现有单元测试、集成测试和 golden test 命令；
4. 确认现有输出 schema 和运行元数据；
5. 生成 `docs/cpicann_baseline_before_xdecomposer.md`；
6. 记录当前：
   - Git commit；
   - Python 版本；
   - lockfile 哈希；
   - CPICANN 权重哈希；
   - catalog 哈希；
   - 0/1/3-norm golden baseline；
   - Docker 镜像标签。

### 禁止事项

- 不修改 CPICANN 推理逻辑；
- 不修改预处理；
- 不添加 XDecomposer 依赖；
- 不下载 XDecomposer 权重。

### 自动验收

根据仓库实际工具调整，但至少包括：

```bash
git status --short
git branch --show-current
git rev-parse HEAD

ruff format --check .
ruff check .
mypy src
pytest -q
```

如果真实模型测试需要本地资产，应单独运行并记录：

```bash
pytest -q -m model
```

### 人工验收

项目负责人检查：

- 当前 `main` 是否干净；
- CPICANN 单相模式是否仍可处理 0/1/3-norm；
- 输出文件和原需求一致；
- golden baseline 是否固定；
- 已创建恢复 tag，例如 `v0.1.0-before-xdecomposer`；
- 没有把权重提交到 Git。

### 阶段停止条件

出现以下任一情况，不得进入 XD-1：

- 当前 CPICANN 测试不通过；
- 默认分支混乱；
- 真实权重和 catalog 不可追溯；
- 现有输出存在未解释回归。

### 建议分支和提交

```text
branch: chore/xdecomposer-baseline
commit: docs: freeze CPICANN baseline before XDecomposer integration
```

---

## Phase XD-1：资产、许可和 Manifest 闸门

### 目标

建立真实 checkpoint、预训练权重和 reference bank 的资产治理机制。此阶段仍不做真实推理。

### Codex 工作

1. 创建 `XDecomposerAssetManifest` schema；
2. 规定以下必填字段：
   - model_id；
   - upstream_repo；
   - upstream_commit；
   - source_license；
   - checkpoint_license；
   - dataset_license；
   - Python/PyTorch/CUDA 版本；
   - xrd_length；
   - num_sources；
   - checkpoint path；
   - checkpoint SHA-256；
   - MAE checkpoint path 和 SHA-256；
   - reference bank path 和 SHA-256；
3. 实现 manifest loader 和 hash verifier；
4. 缺失、空值、文件不存在、哈希不匹配必须返回结构化错误；
5. 添加 `.gitignore` 和 Docker ignore 规则；
6. 创建：
   - `models/xdecomposer/README.md`；
   - `models/xdecomposer/manifest.example.yaml`；
   - `docs/xdecomposer_asset_licenses.md`；
7. 默认配置不得自动下载任何资产；
8. 增加 CLI 校验命令，例如：

```bash
cpicann-xrd xdecomposer verify-assets --manifest /path/manifest.yaml
```

### 自动验收

```bash
pytest -q tests/decomposition/test_asset_manifest.py
pytest -q tests/decomposition/test_asset_hashes.py
ruff check .
mypy src
```

测试必须覆盖：

- 合法 manifest；
- 缺字段；
- 非法 SHA-256；
- 文件不存在；
- 哈希不匹配；
- 许可字段为 `UNKNOWN` 时拒绝生产模式；
- example manifest 可解析但不能用于真实启动。

### 人工验收

必须由人确认，Codex 不得代替：

- checkpoint 是否允许使用；
- checkpoint 是否允许放到服务器；
- checkpoint 是否允许再分发；
- reference bank 是否允许再分发；
- MP20/RRUFF 衍生数据的边界；
- 是否需要在产品中加入第三方 notice；
- 是否允许生成包含 COD/结构信息的结果；
- manifest 中的真实 SHA-256 是否由可信人员计算。

### 阶段停止条件

以下任一情况出现，后续只能使用 stub：

- checkpoint 许可未知；
- reference bank 许可未知；
- 无法获得权重；
- SHA-256 无法固定；
- num_sources 或模型配置无法确认。

### 建议分支和提交

```text
branch: feature/xdecomposer-assets
commit: feat: add XDecomposer asset manifest and closed verification
```

---

## Phase XD-2：独立运行环境 Spike

### 目标

证明 XDecomposer 可以在独立容器中导入、启动和检查运行环境，同时不影响 CPICANN。

### Codex 工作

1. 创建独立 `services/xdecomposer_service`；
2. 固定 Python 3.10；
3. 依据上游实际依赖创建独立 lockfile；
4. 审查并处理可疑依赖：
   - `skimage==0.0` 不得未经验证直接沿用；
   - 仅推理不需要的训练依赖应尽量排除；
5. 创建 `Dockerfile.xdecomposer`；
6. 不把真实权重 bake 进镜像；
7. 增加：
   - `/healthz`：进程存活；
   - `/readyz`：环境和资产是否可用；
   - `/info`：Python、PyTorch、CUDA、GPU、模型配置；
8. 添加 `compose.xdecomposer.yaml`，默认 profile 为 opt-in；
9. CPICANN 原 compose 不启用 XDecomposer 时必须保持原行为；
10. 增加资源限制和只读模型挂载。

### 建议健康返回

```json
{
  "status": "not_ready",
  "python": "3.10.x",
  "torch": "2.10.0",
  "cuda_available": false,
  "assets_present": false,
  "manifest_valid": false
}
```

### 自动验收

CPU 环境：

```bash
docker build -f docker/Dockerfile.xdecomposer -t xdecomposer-spike:test .
docker run --rm xdecomposer-spike:test python -m xdecomposer_service.check_environment
```

Compose：

```bash
docker compose -f compose.yaml -f compose.xdecomposer.yaml config
docker compose -f compose.yaml -f compose.xdecomposer.yaml --profile xdecomposer up -d
curl --fail http://localhost:<port>/healthz
curl http://localhost:<port>/readyz
```

无资产时 `/readyz` 应明确返回 not ready，而不是伪装成功。

### 人工验收

运维或开发负责人检查：

- 镜像中没有权重；
- 镜像大小是否可接受；
- CPICANN 镜像是否未被 CUDA/TensorFlow 依赖污染；
- 无 GPU 机器上能否正常报告“不支持/不可用”；
- 有 GPU 机器上驱动、CUDA 和容器运行时是否匹配；
- XDecomposer 容器停止后 CPICANN 功能是否完全正常。

### 阶段停止条件

- 容器无法稳定构建；
- 上游模块只能依赖硬编码绝对路径；
- CUDA 版本无法与目标云主机兼容；
- XDecomposer 服务会导致 CPICANN 服务失败；
- 运行时必须包含不允许使用的资产。

### 建议分支和提交

```text
branch: feature/xdecomposer-runtime
commit: feat: add isolated XDecomposer runtime spike
```

---

## Phase XD-3：产品契约、Stub Backend 和确定性预处理

### 目标

在不依赖真实权重的情况下，把产品层请求、响应、预处理和服务通信接口稳定下来。

### Codex 工作

1. 定义 `XDecomposerRequest`；
2. 定义 `XDecomposerResult`；
3. 定义 `DecomposedComponent`；
4. 定义 `ReferenceMatch`；
5. 定义结构化错误：
   - asset unavailable；
   - backend unavailable；
   - invalid input；
   - preprocessing failed；
   - inference timeout；
   - output shape mismatch；
6. 实现 `xdecomposer-v1` 预处理：
   - 物理 2θ 输入；
   - 排序；
   - 去重；
   - NaN/Inf 处理；
   - 10–80°；
   - 3500 点；
   - max normalization；
   - 全零输入拒绝；
   - 输出 `[1,1,3500]` float32；
7. 保存预处理元数据；
8. 实现 `StubDecompositionBackend`；
9. stub 应产生确定性的两个 component、activity logits、重建谱和残差；
10. 实现主应用到 worker 的 HTTP client；
11. 设置明确的 timeout、重试和错误映射；
12. 不修改 CPICANN 默认流程。

### 请求建议

```json
{
  "sample_id": "sample-001",
  "two_theta": [10.0, 10.02],
  "intensity": [0.0, 12.0],
  "max_sources": 4,
  "activity_threshold": 0.5,
  "reference_top_k": 5,
  "return_component_patterns": false
}
```

### 响应建议

```json
{
  "backend": "xdecomposer",
  "model_id": "stub-xdecomposer-v1",
  "preprocessing_version": "xdecomposer-v1",
  "components": [],
  "reconstruction_error": 0.0,
  "warnings": [],
  "artifacts": {}
}
```

### 自动验收

```bash
pytest -q tests/decomposition/test_schemas.py
pytest -q tests/decomposition/test_preprocessing.py
pytest -q tests/decomposition/test_stub_backend.py
pytest -q tests/decomposition/test_client.py
```

必须检查：

- 同一输入输出哈希一致；
- 输出长度固定为 3500；
- 非单调 2θ 可规范处理；
- 重复 2θ 可规范处理；
- 全零、空输入、非法数值给出结构化错误；
- timeout 和连接失败不会破坏单相服务；
- component 数量不超过 `max_sources`。

### 人工验收

科研人员确认：

- 10–80° 和 3500 点与目标 checkpoint 一致；
- max normalization 与目标推理路径一致；
- 输入超出范围时的裁剪/插值策略合理；
- 分解结果的物理坐标定义明确；
- activity threshold 只是产品配置，不被描述为模型真值。

### 建议分支和提交

```text
branch: feature/xdecomposer-contract
commit: feat: add XDecomposer contracts, preprocessing and stub service
```

---

## Phase XD-4：真实单样品分解适配器

### 目标

使用经过授权、哈希固定的真实资产，完成一次单样品真实分解。此阶段只做 XDecomposer，不串联 CPICANN。

### 前提

- XD-1 资产闸门通过；
- XD-2 容器运行稳定；
- XD-3 契约和 stub 测试通过；
- 真实 manifest 已由人确认；
- 权重仅通过本地只读挂载进入容器。

### Codex 工作

1. 编写上游模型 adapter；
2. 从 manifest 构建模型；
3. 严格加载目标 checkpoint；
4. 校验 state dict 结构和模型配置；
5. 禁用训练模式，使用 `eval()` 和 inference mode；
6. 输出：
   - `component_patterns [B,S,3500]`；
   - `activity_logits [B,S]`；
   - sigmoid activity probability；
7. 对 component 采用确定性展示排序：
   - 建议先按 `is_active`；
   - 再按 estimated intensity mass；
   - 最后按原 slot index；
8. 保留原 slot index；
9. 计算输入与 component sum 的重建误差；
10. 保存：
    - 输入谱；
    - 所有 slot 谱；
    - 活跃 component 谱；
    - 重建谱；
    - 残差谱；
    - 原始 JSON；
11. 记录模型、设备、依赖和耗时。

### 自动验收

Stub 测试仍必须全部通过。

真实模型测试独立标记：

```bash
pytest -q -m xdecomposer_model
```

重复性测试：

```bash
cpicann-xrd xdecomposer decompose \
  --input tests/fixtures/xdecomposer/mixture.xy \
  --manifest /opt/models/xdecomposer/manifest.yaml \
  --output runs/xd-real-1

cpicann-xrd xdecomposer decompose \
  --input tests/fixtures/xdecomposer/mixture.xy \
  --manifest /opt/models/xdecomposer/manifest.yaml \
  --output runs/xd-real-2
```

比较：

- component 数；
- activity probabilities；
- component pattern hashes；
- reconstruction error；
- 模型和资产 hashes。

### 人工验收

科研人员必须查看真实图像，而不是只看测试通过：

1. 原谱和重建谱是否基本一致；
2. 残差是否存在明显系统峰；
3. 活跃 component 是否具有合理非负强度；
4. 是否出现两个 component 重复表示同一谱；
5. component 是否出现明显噪声放大；
6. activity threshold 是否导致合理的相数；
7. component 排序在重复运行时是否稳定；
8. estimated weight 是否仅标记为 approximate；
9. 是否需要加入最小峰强、最小质量或相似 component 合并规则。

### 阶段停止条件

- 权重无法严格加载；
- 输出 shape 与 manifest 不一致；
- 同设备重复运行结果不可接受地漂移；
- 重建误差异常；
- component 出现 NaN/Inf/大面积负值；
- checkpoint 许可仍未确认。

### 建议分支和提交

```text
branch: feature/xdecomposer-real-adapter
commit: feat: add hash-verified real XDecomposer adapter
```

---

## Phase XD-5：分解结果工件和报告

### 目标

把真实或 stub 分解结果变成可审查、可下载、可追溯的标准输出。

### Codex 工作

每个运行至少生成：

```text
run/
├── decomposition_summary.csv
├── decomposition_metadata.json
├── decomposition_report.md
├── observed_xrd.png
├── reconstruction.png
├── residual.png
├── component_overview.png
├── diagnostics.csv
├── result_bundle.zip
└── components/
    ├── component_01.csv
    ├── component_01.png
    ├── component_02.csv
    └── component_02.png
```

`decomposition_summary.csv` 至少包括：

```text
sample_id
component_display_index
original_slot_index
active_probability
is_active
estimated_weight
pattern_sha256
reconstruction_rmse
warning
```

`decomposition_metadata.json` 至少包括：

- 输入哈希；
- 原文件名；
- 预处理协议；
- 输入角度范围；
- XDecomposer 轴和长度；
- model ID；
- upstream commit；
- checkpoint hashes；
- reference bank hash；
- activity threshold；
- max_sources；
- device；
- Python/PyTorch/CUDA；
- Git commit；
- 容器镜像 digest；
- 耗时；
- warnings。

报告必须写明“非定量相含量”提示。

### 自动验收

```bash
pytest -q tests/decomposition/test_artifacts.py
pytest -q tests/decomposition/test_report.py
pytest -q tests/decomposition/test_bundle.py
```

检查：

- 文件完整；
- CSV schema 固定；
- ZIP 内不含权重和 token；
- component 文件名稳定；
- 非活跃 slot 是否按配置保留或隐藏；
- Markdown 中文可读；
- 所有图像可以打开；
- JSON 可由 schema 重新解析。

### 人工验收

- 报告用词是否科学准确；
- 图例能否区分 input/reconstruction/residual；
- activity 和 estimated weight 是否有清晰定义；
- inactive slot 是否不会误导普通用户；
- ZIP 是否包含足够的复现信息；
- 报告中没有把 component 直接命名成具体物相。

### 建议分支和提交

```text
branch: feature/xdecomposer-reporting
commit: feat: add XDecomposer artifacts and decomposition report
```

---

## Phase XD-6：XDecomposer → CPICANN 分相识别编排

### 目标

把每个活跃分解 component 转换为 CPICANN 输入，并返回每个 component 的 Top-K 候选。

### Codex 工作

1. 新建 orchestration service；
2. 原始谱先使用 `xdecomposer-v1`；
3. component 从 XDecomposer 物理轴恢复为 `(two_theta, intensity)`；
4. 每个 component 再进入 `legacy-cpicann-v1`；
5. 不允许对 3500 点数组直接做索引缩放；
6. 调用现有 CPICANN backend；
7. 对每个 component 应用用户统一元素条件，或允许 component 独立条件；
8. 在过滤后的候选 logits 上重新 softmax；
9. 保存 component 的：
   - Top-K；
   - 原始/过滤概率；
   - catalog metadata；
   - 候选数；
   - 过滤条件；
10. XDecomposer 或 CPICANN 某一 component 失败时，不应丢失整个运行的其余结果；
11. 对低 activity、低质量或高残差 component 给出警告；
12. 生成联合报告。

### 联合结果建议

```json
{
  "sample_id": "mix-001",
  "mode": "multiphase-identification",
  "decomposition": {},
  "components": [
    {
      "component_index": 1,
      "activity_probability": 0.96,
      "estimated_weight": 0.61,
      "cpicann": {
        "candidate_count": 42,
        "top_k": []
      }
    }
  ]
}
```

### 自动验收

使用完全已知的 stub component：

```bash
pytest -q tests/integration/test_decompose_then_classify_stub.py
```

使用合成混合谱：

```text
mixture = 0.7 * phase_A_pattern + 0.3 * phase_B_pattern
```

真实模型测试独立运行：

```bash
pytest -q -m xdecomposer_cpicann_model
```

测试应覆盖：

- 两个 component 均得到 Top-K；
- 元素过滤正确；
- 过滤后概率和为 1；
- 某 component 候选为空；
- 某 component 候选不足 K；
- XDecomposer 后端不可用；
- CPICANN 后端不可用；
- 单个 component 失败但运行仍有诊断；
- 3500→物理轴→4500 的转换确定性。

### 人工验收

科研人员对已知混合谱检查：

1. 真实相是否出现在对应 component 的 Top-5；
2. component 与真实单相谱的相关性；
3. 是否存在 A/B component 对调；
4. 是否需要排列匹配后再评价；
5. 元素条件是否会错误排除真实相；
6. CPICANN 置信度是否被明确标记为条件分数；
7. 低质量分解是否会产生虚假的高分类置信度；
8. 联合报告是否让用户误以为已经完成定量分析。

### 阶段停止条件

- 分解 component 无法稳定映射到物理 2θ；
- CPICANN 对分解谱系统性失效；
- component 间重复、交换或残差导致结果不可解释；
- 测试集上真实相长期无法进入 Top-K；
- 用户界面无法表达不确定性。

### 建议分支和提交

```text
branch: feature/xdecomposer-cpicann-pipeline
commit: feat: orchestrate XDecomposer components through CPICANN ranking
```

---

## Phase XD-7：CLI、API 和 Web 的显式可选入口

### 目标

把新能力暴露给用户，但默认仍保持 CPICANN 单相模式。

### CLI

建议命令：

```bash
cpicann-xrd predict ...
cpicann-xrd decompose ...
cpicann-xrd decompose-and-identify ...
cpicann-xrd xdecomposer verify-assets ...
cpicann-xrd xdecomposer doctor ...
```

### API

建议：

```text
POST /v1/predict
POST /v1/decompose
POST /v1/decompose-and-identify
GET  /healthz
GET  /readyz
GET  /capabilities
```

`/capabilities` 返回：

```json
{
  "cpicann": {"available": true},
  "xdecomposer": {
    "enabled": false,
    "available": false,
    "reason": "assets_not_configured"
  }
}
```

### Web

模式选择：

```text
○ 单相物相识别
○ 多相谱图分解
○ 多相分解并识别
```

多相配置：

- 最大 component 数；
- activity threshold；
- 是否显示 inactive slot；
- Top-K；
- include_must；
- allowed_elements；
- 是否返回 component CSV；
- 是否返回参考库匹配。

### Codex 工作

1. 新模式默认关闭或仅在 capability ready 时显示；
2. XDecomposer 不可用时给出明确解释；
3. API 不能把连接错误暴露成内部 traceback；
4. CLI 返回非零退出码和结构化 diagnostics；
5. Web 不重复实现核心业务逻辑；
6. 三个入口调用同一个 orchestration service；
7. 单相入口行为和输出不变；
8. 增加上传大小、文件数和超时限制。

### 自动验收

```bash
pytest -q tests/integration/test_cli_decomposition.py
pytest -q tests/integration/test_api_decomposition.py
pytest -q tests/integration/test_web_decomposition_helpers.py
pytest -q tests/integration/test_capabilities.py
```

必须验证：

- XDecomposer disabled；
- service unavailable；
- assets invalid；
- stub ready；
- CPICANN-only 环境；
- CLI/API/Web 对同一 stub 输入输出一致；
- 原 `/v1/predict` 回归测试不变。

### 人工验收

产品负责人逐项点击：

- 默认页面仍为单相；
- 用户能看懂三种模式区别；
- 不可用状态不会出现“运行后才失败”；
- 上传、运行、查看和下载流程完整；
- component 标签稳定；
- 警告明显；
- 手机或普通浏览器基本可用；
- 所有下载文件命名明确。

### 建议分支和提交

```text
branch: feature/xdecomposer-product-entry
commit: feat: expose opt-in XDecomposer modes across CLI API and Web
```

---

## Phase XD-8：异步任务、资源控制和云部署

### 目标

使耗时的 GPU 分解任务适合云服务器或多人环境。

### 架构

```text
Web/API
   │
   ▼
Job Store / Queue
   │
   ├── CPICANN worker
   └── XDecomposer GPU worker
```

### Codex 工作

1. 增加任务 API：

```text
POST /v1/jobs
GET  /v1/jobs/{job_id}
GET  /v1/jobs/{job_id}/result
GET  /v1/jobs/{job_id}/download
POST /v1/jobs/{job_id}/cancel
```

2. 任务状态：

```text
queued
validating
preprocessing
decomposing
identifying
reporting
succeeded
failed
cancelled
```

3. 增加：
   - 超时；
   - 最大并发；
   - GPU worker 一次加载模型；
   - 任务取消；
   - 结果隔离；
   - 定时清理；
   - 审计日志；
   - 请求 ID；
4. 资产只读挂载；
5. 运行结果写入独立持久化卷或对象存储；
6. API 不直接持有 CUDA 模型；
7. 添加 Caddy/Nginx HTTPS 配置；
8. 明确 CPU-only 与 GPU profile；
9. 加入 readiness：模型真正加载成功才 ready；
10. 添加 Prometheus 风格指标或结构化日志。

### 自动验收

```bash
docker compose -f compose.yaml -f compose.xdecomposer.yaml config
pytest -q tests/integration/test_jobs.py
pytest -q tests/integration/test_job_timeout.py
pytest -q tests/integration/test_job_isolation.py
```

压测至少包括：

- 1 个任务；
- 2 个并发任务；
- 队列超过并发限制；
- worker 重启；
- API 重启；
- GPU unavailable；
- 任务取消；
- 大文件拒绝；
- 结果过期清理。

### 人工验收

运维人员检查：

- 云主机驱动和 CUDA；
- 容器能访问 GPU；
- 模型只加载一次；
- 服务重启后任务状态处理；
- 磁盘占用；
- 结果清理；
- HTTPS；
- 认证；
- 防火墙；
- secret 不在镜像中；
- 日志不包含敏感谱图内容；
- 备份和恢复；
- 成本和空闲 GPU 策略。

### 建议分支和提交

```text
branch: feature/xdecomposer-jobs
commit: feat: add asynchronous decomposition jobs and GPU worker controls
```

---

## Phase XD-9：科学验证

### 目标

证明该工作流在明确的数据边界内具有可解释、可重复的表现。软件测试通过不等于科学验证通过。

### 数据集设计

至少包括：

1. 合成双相谱；
2. 合成三相谱；
3. 不同比例：
   - 90/10；
   - 70/30；
   - 50/50；
   - 30/70；
4. 少量杂相；
5. 不同噪声；
6. 背景变化；
7. 峰位小幅偏移；
8. 峰宽变化；
9. 实验多相样品；
10. CPICANN 类别库外组分；
11. 元素条件正确；
12. 元素条件错误；
13. 单相输入交给多相模式；
14. 相数超过 checkpoint `num_sources`；
15. 非晶背景明显的输入。

### 指标

#### 分解层

- active source count accuracy；
- reconstruction RMSE；
- component–target correlation；
- permutation-invariant component matching；
- spectral angle/cosine similarity；
- estimated weight MAE；
- false active slot rate；
- missed minor phase rate。

#### 识别层

- component Top-1 accuracy；
- component Top-5 recall；
- 真实相进入候选集比例；
- 元素过滤前后召回率；
- 库外样品的错误高置信率；
- 分解质量与 CPICANN confidence 的相关关系。

#### 性能层

- model cold-start；
- 单样品时延；
- 10 样品批处理；
- GPU 显存；
- CPU 内存；
- 并发吞吐；
- 结果工件大小。

### Codex 工作

1. 只负责构建可复现的评测框架；
2. 实现 permutation-invariant 匹配；
3. 输出原始指标 CSV；
4. 生成评测图；
5. 生成 `docs/xdecomposer_scientific_validation.md` 草稿；
6. 不自行宣布科学结论；
7. 保留失败案例；
8. 不通过删除异常样品提高指标。

### 人工验收

必须由材料/XRD 专业人员完成：

- 样品真值是否可靠；
- 合成方式是否符合物理意义；
- 混合强度比例是否能代表真实相比例；
- 指标阈值是否合理；
- 少量相检测是否可接受；
- 未知相是否会被错误识别；
- 是否允许向用户显示 estimated weight；
- 哪些输入条件必须给出警告；
- 是否具备公开发布或仅内部试用条件。

### 发布闸门建议

在没有足够验证前：

```text
status = experimental
```

只有科学负责人签署后才能标记：

```text
status = validated-for-defined-domain
```

### 建议分支和提交

```text
branch: test/xdecomposer-scientific-validation
commit: test: add reproducible XDecomposer scientific validation suite
```

---

## Phase XD-10：发布与运维交付

### 目标

发布独立、可回滚、明确标记实验状态的 `v0.2.0`。

### Codex 工作

1. 更新 README；
2. 更新 CHANGELOG；
3. 更新 NOTICE；
4. 更新第三方许可；
5. 添加：
   - 安装说明；
   - CPU/GPU profile；
   - 资产挂载说明；
   - 云部署说明；
   - 故障排查；
   - 备份和恢复；
6. CI：
   - 默认不需要真实权重；
   - stub 测试每个 PR 运行；
   - 真实模型 smoke test 仅手工/受保护环境运行；
   - secret 不对外部 PR 暴露；
7. 构建两个镜像：
   - CPICANN app；
   - XDecomposer worker；
8. 镜像使用 immutable tag 和 digest；
9. 生成 SBOM；
10. 对镜像做漏洞扫描；
11. 生成 Release notes；
12. 发布回滚说明。

### 自动验收

```bash
ruff format --check .
ruff check .
mypy src
pytest -q
docker compose config
docker build ...
```

发布前检查：

```text
无真实权重
无 .env
无 token
无用户运行数据
无未授权 reference bank
无内部绝对路径
```

### 人工验收

项目负责人：

- 验收功能范围；
- 确认实验性提示；
- 确认许可；
- 确认科学验证结论；
- 确认云端安全；
- 确认回滚；
- 确认版本号和镜像 digest；
- 在空白服务器完整部署一次；
- 使用一个单相和一个多相样品完成端到端验收。

### 建议标签

```text
v0.2.0-rc1
v0.2.0
```

---

# 7. 人工验收责任矩阵

| 内容 | Codex | 软件负责人 | XRD/材料专家 | 运维负责人 |
|---|---:|---:|---:|---:|
| 代码结构 | 实现 | 验收 | 了解 | 了解 |
| 单元测试 | 实现 | 验收 | — | — |
| 资产 hash | 实现校验 | 复核 | — | 复核 |
| checkpoint 许可 | 不可决策 | 组织确认 | 参与 | 参与 |
| 输入预处理 | 实现 | 复核 | 终审 | — |
| component 科学合理性 | 输出数据 | 参与 | 终审 | — |
| estimated weight 命名 | 实现文案 | 参与 | 终审 | — |
| CPICANN Top-K 解释 | 实现 | 参与 | 终审 | — |
| GPU/CUDA | 实现配置 | 参与 | — | 终审 |
| 云端安全 | 实现配置 | 参与 | — | 终审 |
| 正式发布 | 准备产物 | 终审 | 科学签署 | 运维签署 |

---

# 8. Codex 总控提示词

每个阶段开始时可使用下面的统一模板：

```text
你正在仓库 CPICANN-XRD-App 中工作。

先阅读：
1. AGENTS.md；
2. DEVELOPMENT_PLAN.md；
3. docs/xdecomposer_feasibility.md；
4. docs/xdecomposer_integration_plan.md；
5. 当前阶段涉及的现有代码和测试。

当前只执行 Phase XD-N：<阶段名称>。

第一轮只读预检，不得修改任何文件，不得安装依赖，不得下载权重，
不得提交或推送。请输出：

1. 当前 Git 分支、commit 和工作区状态；
2. 与本阶段有关的现有文件；
3. 拟新增、拟修改、明确不修改的文件；
4. 数据和接口契约；
5. 风险和阻塞项；
6. 将运行的测试和验收命令；
7. 本阶段完成定义。

获得确认后再实施。

实施要求：
- 只做本阶段范围；
- 保持 CPICANN 默认行为不变；
- 不提交任何真实权重、数据集、reference bank、token 或 .env；
- 不把 XDecomposer 依赖安装进 CPICANN 主环境；
- 新代码必须有类型、测试和结构化错误；
- 未实际运行的测试不得声称通过；
- 完成后给出修改文件、命令输出摘要、剩余风险和人工验收清单；
- 不自动 git commit 或 git push，除非得到明确授权。
```

---

# 9. 每阶段 Codex 完成报告格式

要求 Codex 每阶段最后严格使用：

```markdown
## Phase XD-N 完成报告

### 已完成
- ...

### 修改文件
- `path`: 修改原因

### 未修改的关键路径
- ...

### 执行命令
- `...`

### 测试结果
- Passed:
- Failed:
- Skipped:
- Not run:

### 资产与安全检查
- 权重是否进入 Git：
- token 是否进入 Git：
- manifest 是否通过：
- hash 是否验证：

### 已知限制
- ...

### 阻塞项
- ...

### 需要人工验收
1. ...

### 建议下一步
- ...
```

---

# 10. Git 分支和 PR 顺序

推荐一阶段一个 PR：

```text
main
├── chore/xdecomposer-baseline
├── feature/xdecomposer-assets
├── feature/xdecomposer-runtime
├── feature/xdecomposer-contract
├── feature/xdecomposer-real-adapter
├── feature/xdecomposer-reporting
├── feature/xdecomposer-cpicann-pipeline
├── feature/xdecomposer-product-entry
├── feature/xdecomposer-jobs
└── test/xdecomposer-scientific-validation
```

合并顺序必须与阶段顺序一致。

每个 PR 应包含：

- 目标；
- 非目标；
- 接口变化；
- 测试；
- 安全和许可影响；
- 人工验收证据；
- 截图或结果工件；
- 回滚方式。

禁止把 XD-1 到 XD-10 合并成一个超大 PR。

---

# 11. 最终 Definition of Done

只有下面所有条件满足，才能说“XDecomposer 已完成集成”。

## 工程

- [ ] CPICANN 单相流程无回归；
- [ ] XDecomposer 独立容器运行；
- [ ] 资产 manifest 和 SHA-256 校验；
- [ ] 真实权重未进入 Git/公开镜像；
- [ ] stub 测试在普通 CI 中运行；
- [ ] 真实模型 smoke test 在受保护环境运行；
- [ ] CLI/API/Web 调用同一业务层；
- [ ] 异步任务和超时可用；
- [ ] readiness 反映真实模型状态；
- [ ] 输出可复现。

## 产品

- [ ] 三种模式区分清楚；
- [ ] 分解图、重建图、残差图可查看；
- [ ] 每个 component 可下载；
- [ ] 每个 component 有 CPICANN Top-K；
- [ ] activity、estimated weight 和 CPICANN confidence 分字段展示；
- [ ] 错误和后端不可用有清晰提示；
- [ ] ZIP 含完整报告和元数据。

## 科学

- [ ] checkpoint 配置已确认；
- [ ] 3500 点预处理与 checkpoint 一致；
- [ ] component 输出可重复；
- [ ] 已知混合谱有验证；
- [ ] permutation-invariant 评价已实现；
- [ ] 少量相和库外相风险已评估；
- [ ] estimated weight 的用户表述由专家批准；
- [ ] 报告明确非 Rietveld 定量；
- [ ] 适用范围和不适用范围已写入文档。

## 法务与运维

- [ ] 源码许可保留；
- [ ] checkpoint 许可确认；
- [ ] 数据库/reference bank 许可确认；
- [ ] 第三方 notice 完整；
- [ ] HTTPS 和认证；
- [ ] 云端 secret 管理；
- [ ] 备份、清理、监控和回滚；
- [ ] 空白服务器部署验收通过。

---

# 12. 最终建议

当前可行性文档足以支持进入集成开发，但应按以下顺序推进：

```text
先冻结 CPICANN 基线
    ↓
资产与许可闸门
    ↓
独立 XDecomposer 容器
    ↓
产品契约 + stub
    ↓
真实单样品分解
    ↓
分解报告
    ↓
XDecomposer → CPICANN 串联
    ↓
CLI/API/Web
    ↓
异步任务与云部署
    ↓
科学验证
    ↓
v0.2.0 发布
```

**当前最重要的下一步不是编写模型推理代码，而是完成 XD-0 和 XD-1。**

尤其要先解决：

1. 权重和 reference bank 的许可；
2. checkpoint 的文件和 SHA-256；
3. 实际 `num_sources`；
4. Python/PyTorch/CUDA 运行组合；
5. 目标云服务器 GPU 驱动能力。

在这些信息未确认前，Codex 可以完成 manifest、stub、接口、测试和独立容器骨架，但不得把功能标记为真实可用。
