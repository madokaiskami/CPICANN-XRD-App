# CPICANN Baseline Before XDecomposer

日期：2026-07-15

## 目的

本文档冻结引入 XDecomposer 之前的 CPICANN 单相识别基线，作为后续
XD-1 到 XD-10 的回归对照。XD-0 不引入 XDecomposer 依赖，不修改
CPICANN 推理、预处理、CLI、Web、API 或报告路径。

## Git 基线

| 项目 | 值 |
| --- | --- |
| 当前工作分支 | `chore/xdecomposer-baseline` |
| 创建来源 | `origin/phase/00-bootstrap` |
| 当前 commit | `d15af3be488c3373ea0e9423eb41ba0ccdcd1d87` |
| 上游跟踪分支 | `origin/phase/00-bootstrap` |
| 远程 HEAD | `origin/phase/00-bootstrap` |
| 远程 `main` | 不存在；`git fetch origin main` 返回 `couldn't find remote ref main` |

说明：用户要求从 `main` 创建新分支，但当前远程仓库没有 `main` 引用。
`origin/phase/00-bootstrap` 是远程 HEAD，且该分支历史包含 phase 01-11
以及 XDecomposer 计划文档提交。因此 XD-0 当前以该远程 HEAD 作为实际
基线来源。

## 运行环境基线

| 项目 | 值 |
| --- | --- |
| Python | `3.11.15` |
| Platform | `Linux-7.0.0-27-generic-x86_64-with-glibc2.43` |
| PyTorch | `2.13.0+cpu` |
| CUDA available | `False` |
| `uv.lock` SHA-256 | `940568b075eca359d11efe7d3b940c645e9320770686f5f49c00a8775221e7e7` |

## CPICANN 模型与 catalog 基线

来源文件：`configs/models/cpicann-single-d1.yaml`

| 项目 | 值 |
| --- | --- |
| model ID | `cpicann-single-d1` |
| backend | `cpicann` |
| source revision | `3dbfaeab51d272e013d211c7f957760b46ab41cc` |
| architecture | `CPICANN(embed_dim=128,nhead=8,layers=6)` |
| preprocessing | `legacy-cpicann-v1` |
| input points | `4500` |
| classes | `23073` |
| original checkpoint path | `models/cpicann-single-d1/CPICANNsingle_phase_D1.pth` |
| original checkpoint SHA-256 | `d2e898bb4b7482cd7b14953feac437b053617815746f025a6b88ca014e51be98` |
| runtime state-dict path | `models/cpicann-single-d1/CPICANNsingle_phase_D1.state_dict.pth` |
| runtime state-dict SHA-256 | `0f3a452da5218df46eaa37f7d0dadb388e08cdeafa6e24e9d6e2e93512bb2e9a` |
| catalog path | `data/catalog/cpicann_single_phase_d1_catalog.csv` |
| catalog SHA-256 | `393fd648778c2f788efed0d29141051d4214f89152e46ef262e7213bc164e51f` |

`data/catalog/catalog_manifest.json` 基线：

| 项目 | 值 |
| --- | --- |
| manifest SHA-256 | `56ce5883e0e167dfea8ce3c960997ae8bdda42c93c0495857fb5a64e2e4b0f04` |
| schema version | `1.0` |
| source catalog path | `data/catalog/CPICANN_strucs_catalog.csv` |
| source catalog SHA-256 | `749ccde48466588811c00adcd6f382f28ab27e0eec218140dec33a98ab3e0c0a` |
| record count | `23073` |

## Golden Baseline

来源文件：`tests/golden/real_model_samples.json`

| 项目 | 值 |
| --- | --- |
| golden file SHA-256 | `19291e505ad212e73cb72fd101a699174c62f2c3c233f45b04b62c0a646d63f9` |
| samples | `0-norm.txt`, `1-norm.txt`, `3-norm.txt` |
| sample directory | `samples/CPICANN识别` |
| cases per sample | unfiltered; `include_must={Zr,O}` + `allowed_elements={Li,Zr,O}` |
| tolerances | raw logit `1e-5`; probability `1e-8`; confidence `1e-8` |

Golden 样品摘要：

| sample | input SHA-256 | preprocessed SHA-256 | rows | unfiltered Top-1 | filtered Top-1 |
| --- | --- | --- | ---: | --- | --- |
| `0-norm.txt` | `d87bb3165b00d2f3d4963bb86913f90d6a6c72139c8acf94d2a3ea3704b8f490` | `1be59cd3854ec938fa51cec4a76bee01fe0a9240c75929697ee6d8973f098c4f` | 3500 | class `21637`, COD `1511098`, `ErBO3` | class `9828`, COD `1010912`, `ZrO2` |
| `1-norm.txt` | `3bfe9d31c5c96a6421c7d29ae93fec39266645b8f78bc6ad125d7cd1d9e46ebf` | `2e325067f109f5c747bbad611cfc2c3d668346d6ab9be43352b5bbaaba022125` | 3500 | class `15237`, COD `1531376`, `Ta4.002Mn7.998O18` | class `9828`, COD `1010912`, `ZrO2` |
| `3-norm.txt` | `d1eec60975b66d18a23eb30942f74ed2fb83d306029936923e9ea90e259547b2` | `de58e6d8277bd5da70658ac9986a99f02859c91008ca3244763a895ccc269d86` | 3500 | class `11272`, COD `1511679`, `La2Re3B7` | class `9828`, COD `1010912`, `ZrO2` |

完整 Top-5、logit、unfiltered probability、filtered confidence、元素和空间群
固定在 `tests/golden/real_model_samples.json` 中。

## 当前输出 schema 和元数据

外部 schema 来源：`src/cpicann_xrd/schemas.py`

- `SpectrumData`：原始/解析后的谱图输入。
- `PreprocessingConfig`：当前默认 `legacy-cpicann-v1`，10-80 度，
  4500 点，最大强度归一到 100。
- `FilterSpec`：`include_must` 与 `allowed_elements` 元素约束。
- `ModelInfo`：模型 ID、backend、类别数、权重/catalog 哈希和设备。
- `PhaseRecord`：class index 到 COD、公式、元素、空间群的映射。
- `PredictionItem`：rank、class index、logit、未过滤概率和过滤后条件置信度。
- `SamplePrediction`：单样品预测结果。
- `DiagnosticRecord`：失败或忽略文件诊断。
- `RunMetadata`：run ID、应用、模型、运行时、预处理、过滤、Top-K 和输入哈希。

输出工件来源：

- `src/cpicann_xrd/services/run_context.py`：run ID、样品目录和安全文件名。
- `src/cpicann_xrd/services/batch_runner.py`：批量运行、诊断隔离和输出编排。
- `src/cpicann_xrd/reports/exporters.py`：CSV、JSON、PNG、ZIP 原子写入。
- `src/cpicann_xrd/reports/markdown_report.py`：中文 Markdown 报告。

## Docker 基线

Docker 交付文件：

- `docker/Dockerfile.cpu`
- `compose.yaml`
- `.dockerignore`

历史发布验收文档 `docs/release_validation.md` 记录的本地镜像标签：

- `cpicann-xrd-app:rc1`

本轮 XD-0 使用授权 Docker 环境读取到的本机 CPICANN 镜像：

| image | ID | created |
| --- | --- | --- |
| `cpicann-xrd-app:manual` | `f2c38698ea3b` | `2026-07-13 16:34:51 +0800 CST` |
| `cpicann-xrd-app:rc1` | `4587bf3d9555` | `2026-07-13 16:34:51 +0800 CST` |
| `cpicann-xrd-app:local` | `36715844657e` | `2026-07-13 16:13:35 +0800 CST` |
| `cpicann-xrd-app:test` | `d0b04a5ce2eb` | `2026-07-13 16:13:35 +0800 CST` |

## 已确认未进入 Git 的敏感资产类型

命令：

```bash
git ls-files | grep -E '(\.pth|\.pt|\.ckpt|\.safetensors|\.onnx|^models/|^runs/|^dist/|\.env$)' || true
```

结果为空：当前 Git 跟踪文件中未发现权重、checkpoint、运行目录、构建产物
或 `.env`。

## XD-0 自动验收命令与结果

已运行：

```bash
git status --short
git branch --show-current
git rev-parse HEAD
UV_CACHE_DIR=/tmp/cpicann-uv-cache .venv/bin/uv run ruff format --check .
UV_CACHE_DIR=/tmp/cpicann-uv-cache .venv/bin/uv run ruff check .
UV_CACHE_DIR=/tmp/cpicann-uv-cache .venv/bin/uv run mypy src
UV_CACHE_DIR=/tmp/cpicann-uv-cache .venv/bin/uv run pytest -q -m "not model"
UV_CACHE_DIR=/tmp/cpicann-uv-cache .venv/bin/uv run pytest -q -m model
```

真实模型测试依赖本机授权权重目录 `models/cpicann-single-d1`。权重只在工作区
未跟踪目录中使用，不提交到 Git。

结果：

| 命令 | 结果 |
| --- | --- |
| `git diff --check` | PASS |
| `git status --short` | 仅 `docs/cpicann_baseline_before_xdecomposer.md` 未跟踪 |
| `git branch --show-current` | `chore/xdecomposer-baseline` |
| `git rev-parse HEAD` | `d15af3be488c3373ea0e9423eb41ba0ccdcd1d87` |
| `ruff format --check .` | PASS，`66 files already formatted` |
| `ruff check .` | PASS |
| `mypy src` | PASS，`34 source files` |
| `pytest -q -m "not model"` | PASS，`89 passed, 3 deselected` |
| `pytest -q -m model` | PASS，`3 passed, 89 deselected` |

## 本阶段明确不修改

- CPICANN 网络定义。
- CPICANNBackend 和 FakeBackend。
- 4500 点 `legacy-cpicann-v1` 预处理协议。
- 元素过滤与 masked softmax 逻辑。
- CLI、Web、API、报告输出路径。
- Docker 镜像内容。
- XDecomposer 源码、权重、数据集或依赖。

## 风险和后续人工验收

1. 远程仓库当前没有 `main` 分支；XD 分支基线实际来自 `origin/phase/00-bootstrap`。
2. 尚未创建恢复 tag。建议人工确认后创建 `v0.1.0-before-xdecomposer`。
3. `docs/phase_status.md` 中各 CPICANN 阶段人工验收仍标记为待验收。
4. XDecomposer 真实权重、reference bank 和数据许可仍未进入 XD-1 闸门。
