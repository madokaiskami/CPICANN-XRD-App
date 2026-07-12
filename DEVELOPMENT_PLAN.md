# CPICANN-XRD-App 分阶段开发与验收计划

> 文档版本：1.0  
> 更新日期：2026-07-12  
> 主仓库：`madokaiskami/CPICANN-XRD-App`  
> 上游项目：`WPEM/CPICANN`  
> 主要执行者：Codex  
> 文档状态：可执行基线

---

## 0. 文档用途

本文件是 `CPICANN-XRD-App` 的开发总纲，也是 Codex 的任务路由文件。

Codex 每次只执行一个阶段，不得跨阶段提前开发。每一阶段都必须满足以下条件后才能进入下一阶段：

1. 阶段要求的代码、测试和文档已完成；
2. 本阶段列出的验收命令全部通过；
3. `git diff` 中不存在无关改动；
4. 阶段总结中列出修改文件、测试结果、遗留风险；
5. 人工验收后再合并到 `main`。

**禁止 Codex 一次性实现全部系统。** 本项目必须通过小步提交和阶段验收降低科研代码产品化风险。

---

## 1. 产品目标

将 CPICANN 从研究脚本包装为可开箱即用的 XRD 物相识别应用。用户输入一个或多个谱图文件并配置元素约束后，系统自动完成：

1. 输入文件校验；
2. XRD 谱图解析；
3. CPICANN 兼容预处理；
4. 预训练单相模型加载和推理；
5. 候选相元素过滤；
6. 过滤后重新计算 softmax 条件置信度；
7. 输出 Top-K 候选，默认 Top-5；
8. 生成表格、谱图、诊断、元数据和中文报告；
9. 支持 CLI、Web 和 API；
10. 支持 Docker 交付和可复现运行。

---

## 2. 范围边界

### 2.1 v0.1.0 必须包含

- 单文件推理；
- 目录或多文件批量推理；
- `.txt`、`.csv`、`.xy` 输入；
- `.png`、`.rar` 等不支持文件的忽略和诊断记录；
- CPICANN 单相模型；
- `include_must` 元素约束；
- `allowed_elements` 元素范围约束；
- 过滤后 masked softmax；
- Top-K 输出；
- CSV、JSON、Markdown、PNG 输出；
- CLI；
- Streamlit Web 页面；
- FastAPI 基础接口；
- CPU Docker 镜像；
- 不依赖真实权重的普通 CI；
- 使用真实权重的手工或受保护 smoke test。

### 2.2 v0.1.0 不包含

- 多相含量定量；
- Rietveld 或 Le Bail 精修；
- CPICANN 重新训练；
- 图片 OCR 后自动还原谱图；
- 用户账户、数据库和权限系统；
- 分布式任务队列；
- 云端 GPU 自动扩缩容；
- 把 Top-5 置信度解释为物相含量；
- 自动把 `.png` 当作模型输入。

这些功能只能在 v0.1.0 稳定后另立里程碑。

---

## 3. 不可变业务规则

Codex 不得在未修改本文件并取得人工同意的情况下改变以下规则。

### 3.1 输入规则

- 支持扩展名：`.txt`、`.csv`、`.xy`，大小写不敏感；
- `.png` 只能作为辅助图像或被忽略，不能直接作为 CPICANN 输入；
- 不支持文件不能导致整个批次失败；
- 单个样品失败不能中断其他样品；
- 每个输入文件必须计算 SHA-256；
- 文件内容必须经过数值、空值、角度和长度校验。

### 3.2 元素过滤规则

设候选相元素集合为 `candidate_elements`：

```python
must_ok = include_must.issubset(candidate_elements)
allowed_ok = (
    allowed_elements is None
    or candidate_elements.issubset(allowed_elements)
)
is_valid = must_ok and allowed_ok
```

语义：

| 配置 | 含义 |
|---|---|
| `include_must={Zr,O}` | 候选相必须同时含 Zr 和 O，可包含其他元素 |
| `allowed_elements={Li,Zr,O}` | 候选相不能含 Li/Zr/O 之外的元素，但不要求三种全部存在 |
| 两者组合 | 必须含 Zr/O；Li 可有可无；不能含其他元素 |
| `include_must={Li,Zr,O}` 与同范围 `allowed_elements` | 只保留同时含 Li/Zr/O 的三元候选 |

### 3.3 置信度规则

过滤必须作用于模型 logits，而不是对已 softmax 的概率再次做 softmax。

正确过程：

```python
valid_indices = torch.where(valid_mask)[0]
valid_logits = logits[valid_indices]
filtered_probabilities = torch.softmax(valid_logits, dim=0)
```

必须满足：

- 无过滤时，概率在全部候选上归一化；
- 有过滤时，`filtered_confidence` 在过滤后的候选集合上归一化，和为 1；
- 过滤后候选为空时明确失败，不生成伪结果；
- 候选数小于 Top-K 时返回全部候选并附警告；
- 报告必须说明 `filtered_confidence` 是约束候选集内的条件分数，不是物相含量。

### 3.4 可复现规则

每次运行必须记录：

- 应用版本；
- Git commit；
- 输入文件哈希；
- 权重文件哈希；
- 类别目录哈希；
- 模型 ID；
- 预处理协议版本；
- 过滤条件；
- Top-K；
- Python、PyTorch 和关键依赖版本；
- 操作系统和设备；
- 容器镜像摘要（容器运行时）；
- 运行时间和 run ID。

---

## 4. 技术架构

### 4.1 三层结构

```text
用户交互层
├── CLI / Typer
├── Web / Streamlit
└── API / FastAPI
        │
任务管理层
├── 单文件与批量任务
├── 输出目录和运行上下文
├── 诊断与错误隔离
├── 元数据和归档
└── 结果导出
        │
推理核心层
├── SpectrumReader
├── Preprocessor
├── ModelBackend
├── PhaseCatalog
├── ElementFilter
├── ConfidenceCalculator
└── Predictor
```

核心层不得依赖 Streamlit 或 FastAPI。CLI、Web 和 API 必须调用同一个应用服务，不得复制推理逻辑。

### 4.2 模型后端抽象

必须先定义协议，再接入真实 CPICANN：

```python
class InferenceBackend(Protocol):
    @property
    def model_info(self) -> ModelInfo: ...

    def predict_logits(self, x: torch.Tensor) -> torch.Tensor: ...
```

实现两个后端：

1. `FakeBackend`：确定性输出，用于普通 CI、业务流程和 UI 测试；
2. `CPICANNBackend`：加载真实预训练权重，用于正式推理和受保护 smoke test。

普通 CI 不得要求下载受限权重。

### 4.3 CPICANN 源码接入策略

优先级如下：

1. **首选**：将用户自己的 CPICANN fork 固定到明确 commit/tag，通过 Git dependency 或 submodule 接入；
2. 如果原项目无法作为 Python package 导入，先在 fork 中完成最小 package 化；
3. 如果研究代码仍包含大量导入副作用，可在本仓库建立隔离适配器；
4. 只有在直接依赖不可行并记录 ADR 后，才允许复制最小网络定义；复制时必须保留 MIT 许可证、上游路径、原 commit 和修改说明。

不得把训练脚本、数据集和无关 notebook 整体复制进产品核心包。

### 4.4 建议技术栈

- Python 3.11；
- PyTorch；
- NumPy、Pandas；
- SciPy，仅在确有预处理需求时使用；
- Pydantic v2；
- Typer；
- FastAPI；
- Streamlit；
- Matplotlib；
- pymatgen `Composition`，用于可靠解析化学式；
- pytest、pytest-cov；
- Ruff；
- mypy；
- uv 管理环境和锁文件；
- Docker 和 Docker Compose；
- GitHub Actions。

---

## 5. 目标仓库结构

```text
CPICANN-XRD-App/
├── AGENTS.md
├── DEVELOPMENT_PLAN.md
├── README.md
├── LICENSE
├── NOTICE
├── CITATION.cff
├── CHANGELOG.md
├── CONTRIBUTING.md
├── SECURITY.md
├── pyproject.toml
├── uv.lock
├── Makefile
├── compose.yaml
├── .env.example
├── .gitignore
│
├── .codex/
│   └── config.toml
│
├── .github/
│   ├── ISSUE_TEMPLATE/
│   ├── pull_request_template.md
│   └── workflows/
│       ├── ci.yml
│       ├── model-smoke.yml
│       ├── docker.yml
│       └── release.yml
│
├── configs/
│   ├── default.yaml
│   └── models/
│       └── cpicann-single-d1.yaml
│
├── assets/
│   └── catalog/
│       ├── README.md
│       ├── catalog_manifest.json
│       └── cpicann_single_phase_d1_catalog.parquet
│
├── src/
│   └── cpicann_xrd/
│       ├── __init__.py
│       ├── version.py
│       ├── settings.py
│       ├── schemas.py
│       ├── exceptions.py
│       │
│       ├── core/
│       │   ├── spectrum_io.py
│       │   ├── preprocessing.py
│       │   ├── element_filter.py
│       │   └── confidence.py
│       │
│       ├── model/
│       │   ├── protocol.py
│       │   ├── fake_backend.py
│       │   ├── cpicann_backend.py
│       │   ├── loader.py
│       │   ├── registry.py
│       │   └── manifest.py
│       │
│       ├── catalog/
│       │   ├── loader.py
│       │   ├── builder.py
│       │   └── validator.py
│       │
│       ├── services/
│       │   ├── predictor.py
│       │   ├── batch_runner.py
│       │   └── run_context.py
│       │
│       ├── reports/
│       │   ├── csv_exporter.py
│       │   ├── markdown_report.py
│       │   └── plotting.py
│       │
│       ├── cli.py
│       ├── api/
│       │   └── main.py
│       └── web/
│           └── app.py
│
├── scripts/
│   ├── inspect_upstream.py
│   ├── download_model.py
│   ├── convert_checkpoint.py
│   ├── build_catalog.py
│   └── verify_model_bundle.py
│
├── examples/
│   └── spectra/
│       ├── 0-norm.txt
│       ├── 1-norm.txt
│       └── 3-norm.txt
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── golden/
│   └── fixtures/
│
└── docs/
    ├── adr/
    ├── model_contract.md
    ├── preprocessing_contract.md
    ├── output_schema.md
    └── phase_status.md
```

---

## 6. Git 与 Codex 工作规则

### 6.1 仓库预检

当前仓库可能位于另一个 CPICANN 仓库目录之下。开始开发前必须执行：

```bash
git rev-parse --show-toplevel
git remote -v
git status --short --branch
```

预期 `--show-toplevel` 指向 `CPICANN-XRD-App` 本身。如果指向父级 CPICANN 仓库，应把产品仓库移到独立或同级目录，避免意外形成嵌套仓库。

推荐本地结构：

```text
~/Codes/
├── CPICANN/              # 上游 fork
└── CPICANN-XRD-App/      # 产品仓库
```

### 6.2 分支规则

每个阶段使用单独分支：

```text
phase/00-bootstrap
phase/01-model-contract
phase/02-core-scaffold
phase/03-spectrum-preprocessing
phase/04-model-inference
phase/05-filter-confidence
phase/06-batch-report
phase/07-cli
phase/08-web
phase/09-api
phase/10-container-ci
phase/11-release-validation
```

禁止 Codex 直接在 `main` 上开发。

### 6.3 提交规则

- 一个提交只解决一个逻辑主题；
- 使用 Conventional Commits；
- 不提交模型权重、运行结果、密钥、缓存和虚拟环境；
- 不允许 `git add .` 后不检查暂存区；
- Codex 在提交前必须运行 `git diff --check`；
- Codex 默认不得执行 force push；
- Codex 不得自行合并 PR 或删除远程分支。

示例：

```text
chore: initialize project tooling
feat(io): add robust spectrum readers
feat(model): add CPICANN checkpoint loader
fix(filter): recompute confidence on masked logits
test(golden): add normalized sample inference baseline
```

### 6.4 每个阶段的 Codex 输出格式

Codex 完成阶段任务时必须报告：

```text
阶段：Phase N
状态：PASS / BLOCKED

完成内容：
- ...

修改文件：
- ...

执行命令：
- ...

测试结果：
- ...

未完成或风险：
- ...

建议人工检查：
- ...
```

---

## 7. AGENTS.md 基线内容

Phase 0 必须在仓库根目录创建 `AGENTS.md`。内容至少包含以下规则：

```markdown
# AGENTS.md

## Project mission
Build a reproducible, offline-capable XRD phase-identification application around the pretrained CPICANN single-phase model.

## Source of truth
- Read DEVELOPMENT_PLAN.md before starting work.
- Execute only the phase explicitly requested by the user.
- Do not start a later phase until current acceptance commands pass.
- Business rules in DEVELOPMENT_PLAN.md are immutable unless the user approves a change.

## Architecture constraints
- Keep inference core independent from CLI, FastAPI, and Streamlit.
- CLI, API, and Web must call the same service layer.
- Use an InferenceBackend protocol with FakeBackend and CPICANNBackend.
- Never duplicate softmax/filtering logic across interfaces.
- Never treat PNG images as spectra.
- Never interpret filtered confidence as phase fraction.

## Model and data constraints
- Do not commit model weights, secrets, raw licensed datasets, or generated run directories.
- Pin model source revision and verify SHA-256.
- Keep model weights and class catalog version-locked.
- Ordinary CI must run without real pretrained weights.
- Real-model tests must be marked and skipped unless the model is configured.

## Quality gates
Before claiming completion, run the commands required by the current phase, plus:
- git diff --check
- uv run ruff check .
- uv run pytest -q for the available test suite

## Coding conventions
- Python 3.11.
- Type annotations on public functions.
- Pydantic models for external schemas and configuration.
- pathlib instead of string path manipulation.
- Structured exceptions with stable error codes.
- No bare except.
- No silent data correction: every correction or ignored input must be diagnosable.
- User-facing output and reports are Chinese; code identifiers and docstrings may be English.

## Change discipline
- Keep changes limited to the requested phase.
- Do not reformat unrelated files.
- Do not commit or push unless explicitly asked.
- Summarize files changed, commands run, results, and remaining risks.
```

后续如 Codex 重复犯同一错误，应把纠正规则写入最靠近相关代码目录的 `AGENTS.md`，但不得堆积无关长文。

---

# 8. 分阶段实施计划

## Phase 0：仓库基线与开发工具

### 目标

建立可安装、可测试、可持续迭代的空应用骨架，不接入真实模型。

### 任务

1. 检查仓库是否错误嵌套；
2. 创建基础目录；
3. 创建 `pyproject.toml`；
4. 配置 Python 3.11；
5. 配置 uv、Ruff、mypy、pytest、coverage；
6. 创建 `AGENTS.md`；
7. 创建 `.gitignore`、`.env.example`；
8. 创建 `Makefile`；
9. 创建最小包和版本命令；
10. 创建 `docs/phase_status.md`；
11. 创建基础 GitHub Actions CI；
12. 编写 README 的开发环境安装部分。

### 最小 CLI

```bash
uv run cpicann-xrd --version
```

必须输出应用版本并以退出码 0 结束。

### 验收命令

```bash
uv sync --all-extras
uv run cpicann-xrd --version
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest -q
uv run python -m build
git diff --check
```

### 验收标准

- 所有命令通过；
- 至少有一个基础测试；
- wheel 和 sdist 可构建；
- `.gitignore` 明确排除 `*.pth`、`*.pt`、`*.ckpt`、`models/`、`runs/`、`.env`；
- CI 不下载模型权重。

### Codex 启动提示词

```text
阅读 DEVELOPMENT_PLAN.md 和 AGENTS.md。只执行 Phase 0，不得开始 Phase 1。
完成仓库基线、工具配置和最小可安装包。运行 Phase 0 的全部验收命令。
不要提交或推送。最后按阶段输出格式汇报。
```

---

## Phase 1：上游审计与模型契约

### 目标

在编写真实推理代码前，把 CPICANN 的输入、输出、权重和类别映射转化为明确、可验证的合同。

### 任务

1. 审计用户 fork 和上游 CPICANN；
2. 记录使用的上游 URL、branch、commit、许可证；
3. 找到单相模型网络定义；
4. 找到 checkpoint 加载方式；
5. 确认模型期望输入形状、点数、角度范围、强度缩放；
6. 确认输出类别数和类别索引来源；
7. 确认 `0-norm.txt`、`1-norm.txt`、`3-norm.txt` 的读取行为；
8. 确认权重实际文件名、来源、访问要求和哈希获取方式；
9. 决定采用 Git dependency、submodule 或最小复制；
10. 创建 ADR；
11. 创建 `docs/model_contract.md` 和 `docs/preprocessing_contract.md`；
12. 编写只读检查脚本 `scripts/inspect_upstream.py`。

### 必须回答的问题

`docs/model_contract.md` 必须明确：

- 输入 tensor 的 dtype；
- 输入 tensor 的 shape；
- 2θ 范围；
- 固定点数；
- 强度缩放；
- 模型输出是否为 logits；
- 单相类别数；
- checkpoint 顶层键；
- `state_dict` 键是否需要前缀转换；
- 类别索引到 COD 信息的映射来源；
- CPU 推理是否可用；
- 真实权重是否允许再分发；
- 产品需要锁定的上游 commit。

### 禁止事项

- 不得凭记忆写模型参数；
- 不得开始实现 Web；
- 不得在没有映射证据时假定类别行号；
- 不得把权重提交到 Git；
- 不得修改上游研究代码来掩盖不清楚的模型契约。

### 验收命令

```bash
uv run python scripts/inspect_upstream.py --help
uv run pytest -q tests/unit/test_upstream_contract.py
uv run ruff check .
uv run mypy src scripts
git diff --check
```

### 人工验收点

人工必须审核：

- 模型接入 ADR；
- 权重来源和许可说明；
- 类别索引映射证据；
- 预处理合同；
- 是否需要修改 CPICANN fork。

未通过人工验收不得进入 Phase 2。

### Codex 启动提示词

```text
只执行 DEVELOPMENT_PLAN.md 的 Phase 1。
先审计代码和模型资产，输出模型合同与 ADR，不实现正式推理。
所有无法从源码、权重或数据确认的事项必须标记为 UNKNOWN，不得猜测。
运行本阶段验收命令，不要提交或推送。
```

---

## Phase 2：核心类型、配置与 FakeBackend

### 目标

建立稳定的数据模型和业务接口，使后续开发不依赖真实权重。

### 任务

1. 定义 Pydantic 模型：
   - `SpectrumData`；
   - `PreprocessingConfig`；
   - `FilterSpec`；
   - `ModelInfo`；
   - `PhaseRecord`；
   - `PredictionItem`；
   - `SamplePrediction`；
   - `DiagnosticRecord`；
   - `RunMetadata`；
2. 定义稳定错误码；
3. 定义 `InferenceBackend` Protocol；
4. 实现确定性 `FakeBackend`；
5. 定义模型 registry 和 manifest schema；
6. 定义配置加载顺序：CLI 参数 > 环境变量 > 配置文件 > 默认值；
7. 创建最小服务层接口；
8. 编写序列化和 schema 测试。

### 错误码基线

至少包含：

```text
UNSUPPORTED_EXTENSION
EMPTY_FILE
NO_VALID_NUMERIC_ROWS
INVALID_COLUMN_COUNT
NON_FINITE_VALUES
INVALID_ANGLE_RANGE
INSUFFICIENT_ANGLE_COVERAGE
PREPROCESSING_FAILED
MODEL_NOT_INSTALLED
MODEL_HASH_MISMATCH
CATALOG_HASH_MISMATCH
MODEL_OUTPUT_SIZE_MISMATCH
NO_CANDIDATES_AFTER_FILTER
INFERENCE_FAILED
REPORT_GENERATION_FAILED
```

### FakeBackend 要求

- 输入相同则 logits 完全相同；
- 不依赖随机种子；
- 可配置类别数；
- 可针对测试注入指定 logits；
- 不冒充真实模型；
- 结果 metadata 必须标记 `backend=fake`。

### 验收命令

```bash
uv run pytest -q tests/unit/test_schemas.py
uv run pytest -q tests/unit/test_fake_backend.py
uv run pytest -q tests/unit/test_settings.py
uv run ruff check .
uv run mypy src
uv run pytest -q
```

### 验收标准

- 所有外部输入均经过 Pydantic 校验；
- schema 可稳定导出 JSON；
- FakeBackend 能驱动一次最小预测；
- 核心包不导入 Streamlit/FastAPI。

### Codex 启动提示词

```text
只执行 Phase 2。建立领域模型、配置、错误码、InferenceBackend 和 FakeBackend。
不要实现真实 CPICANN，不要实现 Web/API。
优先通过单元测试固定接口。运行本阶段全部验收命令。
```

---

## Phase 3：谱图读取与预处理

### 目标

可靠读取 `.txt`、`.csv`、`.xy`，并生成符合模型合同的固定长度输入。

### 任务

1. 实现扩展名识别；
2. 自动处理逗号、空格、Tab 分隔；
3. 支持有表头和无表头；
4. 兼容 CPICANN 已验证的两列和三列格式；
5. 记录非法行；
6. 处理排序、重复角度、NaN、Inf；
7. 建立 `legacy-cpicann-v1` 预处理协议；
8. 必要时建立 `robust-v1`，但默认不能替换 legacy；
9. 插值到模型合同指定点数；
10. 生成预处理诊断信息；
11. 保存预处理后数据的可选导出结构；
12. 加入 `0-norm.txt`、`1-norm.txt`、`3-norm.txt` fixture。

### 决策规则

- 任何自动修正都必须记录；
- 不得静默删除大量行；
- 角度覆盖不足时必须失败或明确警告，由配置决定；
- 预处理协议必须带版本；
- 不得因为“看起来更合理”而改变 legacy 行为。

### 单元测试矩阵

| 场景 | 预期 |
|---|---|
| 两列 TXT | 成功 |
| 三列 TXT | 按合同计算强度 |
| CSV 有表头 | 成功 |
| CSV 无表头 | 不丢第一行 |
| Tab 分隔 XY | 成功 |
| 大写扩展名 | 成功 |
| PNG | ignored/unsupported |
| RAR | ignored/unsupported |
| 空文件 | `EMPTY_FILE` |
| 非数值文件 | `NO_VALID_NUMERIC_ROWS` |
| 倒序角度 | 排序并记录 |
| 重复角度 | 聚合并记录 |
| NaN/Inf | 按合同失败或清理并记录 |
| 点数不足 | 插值或按覆盖策略失败 |
| 输出长度 | 精确等于模型合同点数 |

### 验收命令

```bash
uv run pytest -q tests/unit/test_spectrum_io.py
uv run pytest -q tests/unit/test_preprocessing.py
uv run pytest -q tests/golden/test_preprocessing_golden.py
uv run ruff check .
uv run mypy src
uv run pytest -q
```

### Golden 要求

对 `0-norm.txt`、`1-norm.txt`、`3-norm.txt` 保存：

- 原始有效行数；
- 角度最小值和最大值；
- 预处理后 shape；
- dtype；
- 输出数组 SHA-256 或稳定数值摘要；
- 警告列表。

### Codex 启动提示词

```text
只执行 Phase 3。实现并测试谱图读取和预处理。
legacy-cpicann-v1 必须严格遵循 Phase 1 的预处理合同。
所有自动修正均产生诊断；PNG 和 RAR 不得进入模型流程。
不要实现真实模型、Web 或 API。
```

---

## Phase 4：真实 CPICANN 模型加载与推理

### 目标

在不破坏普通 CI 的前提下，加载固定版本预训练权重并返回 logits。

### 任务

1. 实现模型 manifest；
2. 实现模型目录和环境变量；
3. 实现权重下载脚本；
4. 固定模型 source revision；
5. 实现 SHA-256 验证；
6. 实现 CPU 默认加载；
7. 可选支持 CUDA；
8. 优先使用安全的 `weights_only=True`；
9. 如旧 checkpoint 不兼容，创建受控转换脚本；
10. 正式运行优先加载纯 `state_dict` 或 safetensors；
11. 实现 `CPICANNBackend`；
12. 验证输出类别数；
13. 将真实模型测试标记为 `model`；
14. 添加 `doctor` 检查命令的服务函数。

### 环境变量

```text
CPICANN_MODEL_DIR
CPICANN_MODEL_ID
CPICANN_DEVICE
HF_TOKEN
CPICANN_ALLOW_MODEL_DOWNLOAD
```

### 安全规则

- 不加载未知来源 pickle；
- 下载后先校验哈希再加载；
- 权重下载必须明确用户触发或配置允许；
- token 不写入 metadata、日志或错误信息；
- 权重不得进入 Git、普通构建上下文或公开 Release；
- 模型和 catalog 必须共同验证版本。

### 验收命令

无真实权重：

```bash
uv run pytest -q -m "not model"
uv run cpicann-xrd doctor --backend fake
```

配置真实权重后：

```bash
uv run python scripts/verify_model_bundle.py --model-id cpicann-single-d1
uv run pytest -q -m model tests/integration/test_real_model.py
uv run cpicann-xrd doctor --backend cpicann
```

通用：

```bash
uv run ruff check .
uv run mypy src scripts
```

### 验收标准

- 无权重时给出 `MODEL_NOT_INSTALLED`，不能栈追踪轰炸普通用户；
- 哈希错误时拒绝加载；
- 模型输出 shape 与合同一致；
- 模型在同一设备、同一输入下结果稳定；
- 普通 CI 仍然可通过。

### Codex 启动提示词

```text
只执行 Phase 4。依据已批准的模型合同实现 CPICANNBackend、manifest、哈希校验和受保护模型测试。
不得把权重提交到仓库，不得改变预处理合同。
普通测试必须在没有真实权重时通过。
```

---

## Phase 5：类别目录、元素过滤与置信度重算

### 目标

把模型类别索引可靠映射到物相信息，并实现经过验证的元素约束和条件置信度。

### 任务

1. 建立类别目录构建脚本；
2. 生成 catalog manifest；
3. 验证类别数、索引连续性和唯一性；
4. 解析 COD ID、化学式、约简式、元素、空间群、空间群号；
5. 用 pymatgen 解析化学式；
6. 实现 `include_must`；
7. 实现 `allowed_elements`；
8. 实现组合过滤；
9. 实现 logits mask；
10. 在过滤后的 logits 上 softmax；
11. 输出 global rank 和 filtered rank；
12. 处理空候选和候选少于 K；
13. 编写元素过滤真值表测试；
14. 编写数值稳定性测试。

### `prediction_top5.csv` 字段

```text
sample_id
filtered_rank
global_rank
class_index
cod_id
formula
reduced_formula
elements
space_group
space_group_number
raw_logit
unfiltered_probability
filtered_confidence
candidate_count_after_filter
```

### 必须通过的真值表

候选集合示例：

```text
Li2ZrO3
ZrO2
LiZrTiO4
HfZrO4
Li2O
```

测试至少覆盖：

- `include_must={Zr,O}`；
- `allowed_elements={Li,Zr,O}`；
- 两者组合；
- `include_must={Li,Zr,O}` 与同范围 allowed；
- 空集合；
- 非法元素符号；
- Top-K 大于候选数。

### 数值断言

```python
assert probabilities.sum() == pytest.approx(1.0)
assert all(include_must <= item.elements for item in results)
assert all(item.elements <= allowed_elements for item in results)
```

### 验收命令

```bash
uv run python scripts/build_catalog.py --check-only
uv run pytest -q tests/unit/test_catalog.py
uv run pytest -q tests/unit/test_element_filter.py
uv run pytest -q tests/unit/test_confidence.py
uv run pytest -q tests/integration/test_filtered_prediction.py
uv run ruff check .
uv run mypy src scripts
uv run pytest -q -m "not model"
```

### 人工验收点

随机抽查至少 10 个 `class_index -> COD ID/公式/空间群` 映射，确认与上游数据一致。

### Codex 启动提示词

```text
只执行 Phase 5。实现版本锁定的类别目录、元素过滤和 masked softmax。
不得对概率再次 softmax。必须输出 global_rank 与 filtered_rank。
先写真值表测试，再实现代码。
```

---

## Phase 6：批量任务、输出、诊断和中文报告

### 目标

完成端到端任务管理和标准化输出。

### 任务

1. 实现 run ID；
2. 实现单文件和批量任务；
3. 单样品错误隔离；
4. 创建输出目录；
5. 生成 `summary.csv`；
6. 每样品生成 `prediction_top5.csv`；
7. 生成 `summary_report.md`；
8. 生成 `run_metadata.json`；
9. 生成 `diagnostics.csv`；
10. 生成 `observed_xrd.png`；
11. 可选生成 `preprocessed_xrd.csv`；
12. 生成 ZIP 归档；
13. 原子写入，避免半成品文件；
14. 处理重复文件名；
15. 记录成功、忽略和失败数量。

### 标准输出结构

```text
runs/<run_id>/
├── summary.csv
├── summary_report.md
├── run_metadata.json
├── diagnostics.csv
├── input_manifest.csv
├── result_bundle.zip
└── samples/
    └── <safe_sample_id>/
        ├── prediction_top5.csv
        ├── observed_xrd.png
        └── preprocessed_xrd.csv
```

### `summary.csv` 字段

```text
sample_id
source_filename
status
candidate_count
returned_top_k
top1_cod_id
top1_formula
top1_confidence
...
top5_cod_id
top5_formula
top5_confidence
warning
```

### `diagnostics.csv` 字段

```text
source_filename
status
stage
error_code
message
rows_read
rows_invalid
angle_min
angle_max
ignored_reason
```

### 中文报告章节

1. 运行摘要；
2. 模型和预处理信息；
3. 元素过滤条件；
4. 每个样品 Top-K；
5. 失败和忽略文件；
6. 结果解释限制；
7. 可复现信息。

报告必须包含：

```text
本结果为 CPICANN 单相分类模型在当前元素约束条件下生成的候选物相排序。
过滤后置信度为候选集合内的相对条件分数，不代表实际物相含量，
也不替代后续结构精修和实验专家确认。
```

### 验收命令

```bash
uv run pytest -q tests/unit/test_run_context.py
uv run pytest -q tests/unit/test_report_exporters.py
uv run pytest -q tests/integration/test_batch_run.py
uv run pytest -q tests/golden/test_output_bundle.py
uv run ruff check .
uv run mypy src
uv run pytest -q -m "not model"
```

### 验收标准

使用 FakeBackend 对包含以下文件的目录运行：

```text
0-norm.txt
1-norm.txt
3-norm.txt
observed.png
archive.rar
bad.csv
```

结果必须：

- 三个有效谱图成功；
- PNG、RAR 被忽略并记录；
- bad.csv 失败但不影响其他文件；
- 标准输出全部存在；
- ZIP 可以解压；
- metadata 中没有绝对敏感路径和 token。

### Codex 启动提示词

```text
只执行 Phase 6。完成批量任务、标准输出、诊断、谱图和中文报告。
用 FakeBackend 做完整端到端测试。单文件失败不得中断批次。
不要实现 Web/API。
```

---

## Phase 7：CLI 产品化

### 目标

为研究人员和自动化流程提供稳定命令行接口。

### 命令设计

```bash
cpicann-xrd --version
cpicann-xrd doctor
cpicann-xrd models list
cpicann-xrd models install
cpicann-xrd models verify
cpicann-xrd predict
cpicann-xrd batch
```

### 典型命令

```bash
cpicann-xrd predict \
  --input examples/spectra/0-norm.txt \
  --include-must Zr O \
  --allowed-elements Li Zr O \
  --top-k 5 \
  --output runs
```

```bash
cpicann-xrd batch \
  --input examples/spectra \
  --include-must Zr O \
  --allowed-elements Li Zr O \
  --top-k 5 \
  --output runs
```

### CLI 规则

- 人类输出默认中文；
- 提供 `--json` 机器可读模式；
- 参数错误退出码为 2；
- 任务失败使用非零退出码；
- 部分成功批次必须有稳定退出码并文档化；
- 不打印 token；
- 不默认下载模型，除非用户明确执行 `models install`；
- 支持 `--backend fake` 仅用于测试和演示，并明确标记。

### 验收命令

```bash
uv run cpicann-xrd --help
uv run cpicann-xrd doctor --backend fake
uv run cpicann-xrd predict --backend fake --input examples/spectra/0-norm.txt --include-must Zr O --allowed-elements Li Zr O --top-k 5 --output /tmp/cpicann-test
uv run cpicann-xrd batch --backend fake --input examples/spectra --top-k 5 --output /tmp/cpicann-batch
uv run pytest -q tests/integration/test_cli.py
uv run ruff check .
uv run mypy src
```

配置真实模型后额外执行：

```bash
uv run cpicann-xrd predict --backend cpicann --input examples/spectra/0-norm.txt --include-must Zr O --allowed-elements Li Zr O --top-k 5 --output /tmp/cpicann-real
```

### Codex 启动提示词

```text
只执行 Phase 7。使用 Typer 建立稳定 CLI，所有命令调用现有 service 层。
不要在 CLI 内复制预处理、过滤或报告逻辑。
补充退出码和机器可读输出测试。
```

---

## Phase 8：Streamlit Web 应用

### 目标

提供非程序用户可直接操作的本地 Web 页面。

### 页面功能

- 上传单个或多个 `.txt/.csv/.xy`；
- 展示并记录不支持文件；
- “必须包含元素”多选；
- “允许元素范围”多选；
- Top-K，默认 5；
- 模型状态；
- 运行按钮；
- 每个样品的 Top-K 表格；
- observed XRD 图；
- 诊断提示；
- 下载 `summary.csv`；
- 下载中文报告；
- 下载完整 ZIP。

### UI 文案规则

必须清晰区分：

- 样品名称或名义配比；
- 模型预测物相；
- 过滤前全局概率；
- 过滤后条件置信度；
- 单相候选排序；
- 实际多相含量。

### 工程规则

- Streamlit 不直接加载模型业务细节；
- 使用缓存确保模型只加载一次；
- 上传文件写入临时隔离目录；
- 清理临时文件；
- 设置文件数量和大小限制；
- 页面异常不得显示敏感栈信息；
- 支持 FakeBackend 演示模式；
- UI 测试主要验证服务调用和页面状态，不依赖真实权重。

### 验收命令

```bash
uv run streamlit run src/cpicann_xrd/web/app.py --server.headless true
uv run pytest -q tests/integration/test_web_service.py
uv run pytest -q -m "not model"
uv run ruff check .
uv run mypy src
```

### 人工验收场景

上传：

```text
0-norm.txt
1-norm.txt
3-norm.txt
observed.png
```

选择：

```text
必须包含：Zr, O
允许范围：Li, Zr, O
Top-K：5
```

确认：

- 三个谱图有结果；
- PNG 被清晰提示为不支持输入；
- 表格字段完整；
- 图像显示正常；
- CSV、报告、ZIP 可下载；
- 条件置信度解释清晰。

### Codex 启动提示词

```text
只执行 Phase 8。实现 Streamlit UI，并复用现有 service 层。
不在 UI 中实现任何新的推理或过滤算法。
先使用 FakeBackend 完成自动测试，再进行真实模型人工验收。
```

---

## Phase 9：FastAPI 接口

### 目标

为后续系统集成提供稳定 API，不引入复杂任务队列。

### API 基线

```text
GET  /healthz
GET  /readyz
GET  /v1/models
POST /v1/predict
POST /v1/batch
GET  /v1/runs/{run_id}
GET  /v1/runs/{run_id}/download
```

### API 规则

- OpenAPI schema 可生成；
- 错误结构稳定；
- 请求 ID 和 run ID 可追踪；
- 上传大小受限；
- 文件名安全处理；
- 不暴露本地绝对路径；
- 不接受远程 URL 下载作为 v0.1.0 默认功能；
- API 调用同一 service 层；
- 同步接口适合小批量，超时边界必须文档化；
- 不在 v0.1.0 引入 Celery/Redis。

### 错误响应示例

```json
{
  "error": {
    "code": "NO_CANDIDATES_AFTER_FILTER",
    "message": "元素过滤后没有可用候选相",
    "request_id": "..."
  }
}
```

### 验收命令

```bash
uv run uvicorn cpicann_xrd.api.main:app --host 127.0.0.1 --port 8000
uv run pytest -q tests/integration/test_api.py
uv run python -c "from cpicann_xrd.api.main import app; print(app.openapi()['info']['title'])"
uv run ruff check .
uv run mypy src
```

### Codex 启动提示词

```text
只执行 Phase 9。建立 FastAPI 接口，复用现有 service 层和 Pydantic schema。
不得引入数据库、Celery 或 Redis。补齐文件限制、安全文件名和错误结构测试。
```

---

## Phase 10：Docker、CI/CD 与供应链控制

### 目标

实现可重复构建、CPU 开箱运行和自动质量门禁。

### 任务

1. 创建多阶段 CPU Dockerfile；
2. 可选创建 CUDA Dockerfile，但不阻塞 v0.1.0；
3. 创建 `compose.yaml`；
4. 添加非 root 用户；
5. 创建健康检查；
6. 排除权重和运行输出；
7. 创建 `.dockerignore`；
8. 完善普通 CI；
9. 创建受保护 `model-smoke.yml`；
10. 创建 Docker 构建工作流；
11. 创建 tag release 工作流；
12. 生成 SBOM 或至少依赖清单；
13. 进行依赖漏洞扫描；
14. 固定基础镜像版本或 digest；
15. README 添加三步启动说明。

### Docker 行为

默认启动 Web：

```bash
docker compose up --build
```

CLI：

```bash
docker compose run --rm app cpicann-xrd doctor --backend fake
```

权重采用挂载卷或首次显式安装，不默认打进公开镜像：

```text
./models:/app/models
./runs:/app/runs
```

### GitHub Actions 分工

`ci.yml`：

- format check；
- lint；
- type check；
- unit/integration tests；
- 不使用真实权重。

`model-smoke.yml`：

- 手工触发或主分支受保护触发；
- 使用 secret 获取权重；
- 验证哈希；
- 跑 0/1/3-norm；
- 不上传权重 artifact。

`docker.yml`：

- 构建镜像；
- 运行容器 smoke test；
- PR 不推送；
- main/tag 可推 GHCR。

`release.yml`：

- tag 触发；
- 构建 wheel/sdist；
- 生成校验和；
- 发布源码和不含权重的镜像；
- 生成 release notes。

### 验收命令

```bash
docker build -f docker/Dockerfile.cpu -t cpicann-xrd-app:test .
docker run --rm cpicann-xrd-app:test cpicann-xrd --version
docker run --rm cpicann-xrd-app:test cpicann-xrd doctor --backend fake
docker compose config
docker compose up -d
docker compose ps
docker compose down
uv run pytest -q -m "not model"
```

### Codex 启动提示词

```text
只执行 Phase 10。实现不含权重的 CPU Docker 交付和分层 GitHub Actions。
普通 CI 不得访问模型 secret。容器使用非 root 用户，并通过 FakeBackend smoke test。
```

---

## Phase 11：真实模型 Golden Test、发布候选与验收

### 目标

用真实预训练权重完成科学与工程验收，形成 `v0.1.0-rc1`。

### 任务

1. 固定模型 revision 和 SHA-256；
2. 固定 catalog SHA-256；
3. 对 0/1/3-norm 建立真实模型 golden baseline；
4. 保存未过滤 Top-5；
5. 保存 `include_must={Zr,O}`、`allowed_elements={Li,Zr,O}` 的过滤后 Top-5；
6. 保存候选数量；
7. 设定合理浮点容差；
8. CPU 重复运行；
9. Docker 重复运行；
10. CLI、Web、API 结果一致性测试；
11. 生成发布候选；
12. 完成 README、NOTICE、CITATION、SECURITY；
13. 确认权重再分发边界；
14. 整理已知限制。

### Golden 文件不得包含

- 权重二进制；
- token；
- 受限原始数据；
- 无法公开再分发的完整 catalog 内容。

可以包含：

- 输入 fixture 的合法副本或哈希；
- 预期 class index；
- 预期 COD ID；
- 概率和容差；
- 模型、catalog、代码哈希。

### 最终端到端验收

```bash
uv sync --frozen --all-extras
uv run ruff format --check .
uv run ruff check .
uv run mypy src scripts
uv run pytest -q -m "not model"
uv run pytest -q -m model
uv run python -m build
uv run cpicann-xrd doctor --backend cpicann
uv run cpicann-xrd batch \
  --backend cpicann \
  --input examples/spectra \
  --include-must Zr O \
  --allowed-elements Li Zr O \
  --top-k 5 \
  --output /tmp/cpicann-release-test

docker build -f docker/Dockerfile.cpu -t cpicann-xrd-app:rc1 .
docker run --rm \
  -v "$PWD/models:/app/models:ro" \
  -v "$PWD/examples:/app/examples:ro" \
  -v "/tmp/cpicann-container-test:/app/runs" \
  cpicann-xrd-app:rc1 \
  cpicann-xrd batch \
  --backend cpicann \
  --input /app/examples/spectra \
  --include-must Zr O \
  --allowed-elements Li Zr O \
  --top-k 5 \
  --output /app/runs
```

### v0.1.0 发布门槛

- 所有自动测试通过；
- 真实模型 golden test 通过；
- CLI、Web、API 对相同输入结果一致；
- Docker CPU 运行成功；
- 标准输出完整；
- 权重和 catalog 对应关系经过人工抽查；
- 报告不把置信度表述为含量；
- 未知许可问题已解决或在 README 明确限制；
- 没有 secret、权重、运行结果进入 Git 历史；
- 发布镜像不含受限权重；
- 形成 `v0.1.0-rc1`，人工试用后再打 `v0.1.0`。

### Codex 启动提示词

```text
只执行 Phase 11。使用已授权、已配置的真实权重完成 golden、跨接口一致性和发布候选验收。
不得修改业务规则来迁就失败测试。发现差异时先定位根因并报告。
不得把权重、token 或受限数据提交到 Git。
```

---

# 9. 测试分层

## 9.1 单元测试

覆盖：

- schema；
- 配置；
- 文件解析；
- 预处理；
- 元素解析；
- 过滤；
- softmax；
- catalog；
- exporter；
- metadata；
- 错误码。

特点：快速、无网络、无权重。

## 9.2 集成测试

覆盖：

- FakeBackend 端到端；
- CLI；
- API；
- Web service；
- 批量任务；
- ZIP 输出。

特点：无真实权重，可在普通 CI 运行。

## 9.3 模型测试

标记：

```python
@pytest.mark.model
```

覆盖：

- checkpoint 加载；
- 真实输出 shape；
- 0/1/3-norm 推理；
- 模型和 catalog 匹配；
- golden baseline。

特点：需要权重，只在受保护环境运行。

## 9.4 Golden 测试

Golden 不是为了固化偶然实现，而是固定以下合同：

- legacy 预处理结果；
- 类别映射；
- 过滤语义；
- 输出 schema；
- 已批准模型版本下的预测结果。

任何 golden 更新必须在 PR 中说明原因并人工审核。

---

# 10. 数据与输出契约

## 10.1 PhaseRecord

至少包含：

```text
class_index: int
cod_id: str
formula: str
reduced_formula: str
elements: frozenset[str]
space_group: str | None
space_group_number: int | None
```

## 10.2 SamplePrediction

至少包含：

```text
sample_id
source_filename
status
backend
model_id
preprocessing_version
filter_spec
candidate_count_before_filter
candidate_count_after_filter
requested_top_k
returned_top_k
predictions
warnings
```

## 10.3 DiagnosticRecord

稳定字段不得随意删除，新增字段需保持向后兼容。

## 10.4 run_metadata.json

至少包含：

```json
{
  "schema_version": "1.0",
  "run_id": "...",
  "created_at": "...",
  "application": {
    "version": "...",
    "git_commit": "..."
  },
  "model": {
    "backend": "cpicann",
    "model_id": "cpicann-single-d1",
    "source_revision": "...",
    "weight_sha256": "...",
    "catalog_sha256": "...",
    "num_classes": 23073
  },
  "runtime": {
    "python": "...",
    "torch": "...",
    "device": "cpu",
    "platform": "..."
  },
  "preprocessing": {
    "name": "legacy-cpicann-v1",
    "two_theta_min": 10.0,
    "two_theta_max": 80.0,
    "points": 4500
  },
  "filter": {
    "include_must": ["O", "Zr"],
    "allowed_elements": ["Li", "O", "Zr"]
  },
  "top_k": 5,
  "inputs": []
}
```

其中具体模型参数必须以 Phase 1 审计结果为准；如果上例与模型合同冲突，应修改示例而不是修改合同事实。

---

# 11. Makefile 统一入口

Codex 应优先维护以下命令，避免 README、CI 和本地命令漂移：

```makefile
setup:
	uv sync --all-extras

format:
	uv run ruff format .

lint:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy src scripts

test:
	uv run pytest -q -m "not model"

test-model:
	uv run pytest -q -m model

check: lint test

build:
	uv run python -m build

run-web:
	uv run streamlit run src/cpicann_xrd/web/app.py

run-api:
	uv run uvicorn cpicann_xrd.api.main:app --reload
```

CI 应调用 `make` 目标，而不是维护另一套逻辑。

---

# 12. Codex 使用流程

## 12.1 首次初始化

在产品仓库根目录运行：

```bash
codex
```

进入后先要求：

```text
阅读 DEVELOPMENT_PLAN.md。检查仓库状态和当前阶段，只给出执行计划，不修改文件。
```

确认计划后再发送对应阶段提示词。

## 12.2 每个阶段推荐会话

一个阶段建议分成三个 Codex 任务：

1. **计划任务**：审查需求、列文件和测试，不改代码；
2. **实现任务**：实现本阶段并运行测试；
3. **审查任务**：独立检查 diff、边界情况和回归。

不要让两个并行 Codex 线程修改同一批文件。

## 12.3 通用实现提示词模板

```text
你正在开发 CPICANN-XRD-App。

请先阅读：
1. AGENTS.md
2. DEVELOPMENT_PLAN.md
3. docs/phase_status.md
4. 与当前阶段有关的 ADR 和合同

当前只执行 Phase <N>：<阶段名称>。

要求：
- 不得开始后续阶段；
- 先检查已有实现和测试；
- 优先补测试，再写实现；
- 不改变不可变业务规则；
- 不提交、推送或修改远程仓库；
- 不下载或提交模型权重，除非本阶段明确要求且用户已授权；
- 运行 Phase <N> 的全部验收命令；
- 如果被上游缺失信息阻塞，停止猜测，记录证据、UNKNOWN 项和最小解决建议；
- 最后按 DEVELOPMENT_PLAN.md 的阶段输出格式汇报。
```

## 12.4 通用代码审查提示词模板

```text
作为独立审查者，阅读 AGENTS.md、DEVELOPMENT_PLAN.md 和当前分支 diff。
只审查 Phase <N> 的实现，不添加新功能。
重点检查：
- 是否满足本阶段验收标准；
- 是否违反不可变业务规则；
- 模型与 catalog 是否可能错配；
- 元素集合方向是否写反；
- 是否错误地对概率再次 softmax；
- 是否有静默数据修正；
- 是否泄露 token、绝对路径或权重；
- 是否有 CLI/Web/API 逻辑重复；
- 是否有缺失的失败测试；
- 是否有无关 diff。

先给出按严重程度排序的问题，再给出最小修复方案。没有问题时明确说明验证过哪些命令。
```

## 12.5 阶段结束提示词模板

```text
重新运行本阶段所有验收命令，检查 git diff --check 和 git status。
更新 docs/phase_status.md，但不要修改 DEVELOPMENT_PLAN.md 的需求内容。
列出修改文件、测试命令、通过结果、未解决风险和建议人工检查项。
不要 commit 或 push。
```

---

# 13. 人工验收清单

每个 PR 合并前人工检查：

- [ ] 分支只包含一个阶段；
- [ ] Codex 已阅读 `AGENTS.md`；
- [ ] 验收命令有真实输出；
- [ ] 没有跳过失败测试；
- [ ] 没有修改测试来掩盖错误；
- [ ] 没有提交权重；
- [ ] 没有提交 `.env` 或 token；
- [ ] 没有无关格式化；
- [ ] schema 变化有说明；
- [ ] 新增依赖有必要性说明；
- [ ] 用户可见错误为中文且有稳定错误码；
- [ ] 科学解释没有夸大模型能力；
- [ ] README 与实际命令一致；
- [ ] 当前阶段状态已更新。

---

# 14. 风险登记

| 风险 | 影响 | 处理 |
|---|---|---|
| 权重和源码版本不一致 | 预测错误或无法加载 | manifest 固定 revision 和 SHA-256 |
| catalog 与输出类别错位 | 返回错误物相，属于严重科学错误 | catalog 哈希、长度、索引和人工抽查 |
| 原预处理行为不明确 | 结果与研究脚本不一致 | Phase 1 合同 + legacy golden |
| 权重 pickle 安全风险 | 任意代码执行风险 | 可信来源、哈希、`weights_only`、受控转换 |
| 权重许可不允许再分发 | 公开镜像违规 | 镜像不含权重，用户显式安装或挂载 |
| 普通 CI 依赖受限模型 | PR 无法稳定测试 | FakeBackend + model marker |
| 过滤集合方向写反 | 业务结果错误 | 真值表测试 |
| 对概率再次 softmax | 置信度数学错误 | logits 级测试和代码审查 |
| 自动清洗过度 | 谱图被静默改变 | 诊断记录和覆盖阈值 |
| Web、API、CLI 结果不一致 | 用户无法复现 | 共用 service 层 + 一致性测试 |
| 把 Top-5 当作多相含量 | 科学解释错误 | 报告固定免责声明 |
| 产品仓库嵌套在上游 Git 中 | 提交和 remote 混乱 | Phase 0 仓库预检 |
| Codex 跨阶段大改 | 难以审查和回滚 | 单阶段分支、明确提示词、人工 gate |

---

# 15. 里程碑状态表

Codex 不应自行把未人工验收的阶段标记为完成。

| 阶段 | 内容 | 自动验收 | 人工验收 | 状态 |
|---|---|---:|---:|---|
| Phase 0 | 仓库基线 | 必须 | 必须 | TODO |
| Phase 1 | 模型合同 | 必须 | 必须 | TODO |
| Phase 2 | 核心 schema/FakeBackend | 必须 | 建议 | TODO |
| Phase 3 | 谱图和预处理 | 必须 | 必须 | TODO |
| Phase 4 | 真实模型 | 必须 | 必须 | TODO |
| Phase 5 | catalog/过滤/置信度 | 必须 | 必须 | TODO |
| Phase 6 | 批量和报告 | 必须 | 必须 | TODO |
| Phase 7 | CLI | 必须 | 必须 | TODO |
| Phase 8 | Web | 必须 | 必须 | TODO |
| Phase 9 | API | 必须 | 建议 | TODO |
| Phase 10 | Docker/CI | 必须 | 必须 | TODO |
| Phase 11 | RC 发布验收 | 必须 | 必须 | TODO |

---

# 16. Definition of Done

只有同时满足以下条件，项目才可称为“开箱即用”：

1. 新用户按照 README 在干净 Ubuntu 环境可启动；
2. `docker compose up` 可运行 Web；
3. CLI 可处理单文件和目录；
4. `.txt/.csv/.xy` 可用；
5. `.png/.rar` 不会误入模型；
6. 0/1/3-norm 可完成真实模型推理；
7. 元素约束符合真值表；
8. 过滤后置信度在有效候选上归一化；
9. 空候选和不足 Top-K 有明确提示；
10. 标准输出全部生成；
11. CLI、Web、API 结果一致；
12. 同输入、同权重、同配置可复现；
13. 权重和 catalog 可追溯且不会错配；
14. 普通 CI 不需要模型权重；
15. 公开代码和镜像不泄露受限权重或密钥；
16. 报告明确单相模型和置信度的解释限制；
17. `v0.1.0` 有 tag、changelog 和发布说明。

---

# 17. 参考资料

## CPICANN

- 上游 GitHub：<https://github.com/WPEM/CPICANN>
- 源码与复现资料：<https://huggingface.co/AI4Cryst/CPICANN>
- 预训练模型仓库：<https://huggingface.co/caobin/pretrainCPICANN>
- 论文 DOI：<https://doi.org/10.1107/S2052252524005323>

## Codex

- Codex CLI：<https://developers.openai.com/codex/cli>
- Codex prompting：<https://developers.openai.com/codex/prompting>
- Codex workflows：<https://developers.openai.com/codex/workflows>
- AGENTS.md：<https://developers.openai.com/codex/guides/agents-md>
- Sandbox 与 approvals：<https://developers.openai.com/codex/concepts/sandboxing>

---

## 18. 立即执行的第一条 Codex 指令

在确认 `DEVELOPMENT_PLAN.md` 已位于仓库根目录后，进入仓库运行 Codex，并发送：

```text
阅读 DEVELOPMENT_PLAN.md。现在只做 Phase 0 的只读预检：
1. 检查 Git 顶层目录、remote、分支和工作区；
2. 检查当前文件结构；
3. 对照 Phase 0 列出拟创建和拟修改文件；
4. 列出将运行的验收命令；
5. 指出任何阻塞项。

本轮不得修改文件、不得安装依赖、不得提交或推送。只返回执行计划。
```

人工确认该计划后，再发送 Phase 0 的正式实现提示词。
