# CPICANN-XRD-App

CPICANN-XRD-App 是围绕预训练 CPICANN 单相模型构建的本地 XRD 物相候选排序工具，提供 Web、CLI、API 和 Docker 部署方式。

当前版本为 `v0.2.0-rc1` 发布候选。CPICANN 单相识别仍是默认工作流；XDecomposer 多相功能为显式 opt-in 且保持 `experimental` 状态。

## 快速开始

首次使用只需要完成三件事：

1. 安装依赖。
2. 放置已授权的 CPICANN 权重。
3. 启动 Web 页面。

```bash
git clone https://github.com/madokaiskami/CPICANN-XRD-App.git
cd CPICANN-XRD-App

mamba create -y -p .venv python=3.11 pip
source .venv/bin/activate
python -m pip install uv
UV_CACHE_DIR=/tmp/cpicann-uv-cache .venv/bin/uv sync --frozen --all-extras
```

如果本机已经安装 `uv`，也可以直接运行：

```bash
uv sync --frozen --all-extras
```

## 放置模型权重

本仓库不提交、不打包真实 CPICANN 权重。请把已授权权重放到：

```text
models/cpicann-single-d1/CPICANNsingle_phase_D1.state_dict.pth
```

目录结构应为：

```text
CPICANN-XRD-App/
  models/
    cpicann-single-d1/
      CPICANNsingle_phase_D1.state_dict.pth
```

验证真实模型是否可用：

```bash
UV_CACHE_DIR=/tmp/cpicann-uv-cache .venv/bin/uv run cpicann-xrd doctor --backend cpicann
```

期望输出包含：

```text
状态：ok
后端：cpicann
模型：cpicann-single-d1
```

## 启动 Web

```bash
UV_CACHE_DIR=/tmp/cpicann-uv-cache .venv/bin/uv run streamlit run src/cpicann_xrd/web/app.py
```

浏览器打开：

```text
http://localhost:8501
```

Web 页面只使用真实 `cpicann` 后端。上传 `.txt`、`.csv` 或 `.xy` 谱图后，可设置元素过滤条件并下载结果包。

常用过滤示例：

```text
必须包含元素：Zr, O
允许元素范围：Li, Zr, O
Top-K：5
```

## CLI 使用

批量识别真实样品：

```bash
UV_CACHE_DIR=/tmp/cpicann-uv-cache .venv/bin/uv run cpicann-xrd batch \
  --backend cpicann \
  --input samples/CPICANN识别/0-norm.txt \
  --input samples/CPICANN识别/1-norm.txt \
  --input samples/CPICANN识别/3-norm.txt \
  --include-must Zr O \
  --allowed-elements Li Zr O \
  --top-k 5 \
  --output /tmp/cpicann-release-test
```

输出目录中会包含：

```text
summary.csv
summary_report.md
run_metadata.json
diagnostics.csv
input_manifest.csv
result_bundle.zip
samples/<sample_id>/prediction_top5.csv
samples/<sample_id>/observed_xrd.png
samples/<sample_id>/preprocessed_xrd.csv
```

## API 使用

启动本地 API：

```bash
UV_CACHE_DIR=/tmp/cpicann-uv-cache .venv/bin/uv run uvicorn cpicann_xrd.api.main:app --host 127.0.0.1 --port 8000
```

健康检查：

```bash
curl -fsS http://127.0.0.1:8000/healthz
curl -fsS http://127.0.0.1:8000/readyz
curl -fsS http://127.0.0.1:8000/v1/models
curl -fsS http://127.0.0.1:8000/capabilities
```

上传批量样品：

```bash
curl -fsS -X POST http://127.0.0.1:8000/v1/batch \
  -F "backend=cpicann" \
  -F "include_must=Zr" \
  -F "include_must=O" \
  -F "allowed_elements=Li" \
  -F "allowed_elements=Zr" \
  -F "allowed_elements=O" \
  -F "top_k=5" \
  -F "files=@samples/CPICANN识别/0-norm.txt" \
  -F "files=@samples/CPICANN识别/1-norm.txt" \
  -F "files=@samples/CPICANN识别/3-norm.txt"
```

异步任务 API：

```bash
curl -fsS -X POST http://127.0.0.1:8000/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"mode":"capabilities"}'
```

返回的 `job_id` 可用于：

```text
GET  /v1/jobs/{job_id}
GET  /v1/jobs/{job_id}/result
GET  /v1/jobs/{job_id}/download
POST /v1/jobs/{job_id}/cancel
```

当前 job store 是 in-process API contract 实现；生产多人环境应替换为持久化队列。

## XDecomposer 实验入口

XDecomposer 不会默认启用，也不会自动下载权重。

检查能力状态：

```bash
UV_CACHE_DIR=/tmp/cpicann-uv-cache .venv/bin/uv run cpicann-xrd xdecomposer doctor --json
```

验证本地资产 manifest：

```bash
UV_CACHE_DIR=/tmp/cpicann-uv-cache .venv/bin/uv run cpicann-xrd xdecomposer verify-assets \
  --manifest models/xdecomposer/manifest.yaml \
  --production
```

Stub 合约测试：

```bash
UV_CACHE_DIR=/tmp/cpicann-uv-cache .venv/bin/uv run cpicann-xrd decompose \
  --input examples/spectra/0-norm.txt \
  --xdecomposer-backend stub \
  --json
```

真实 XDecomposer 分解需要授权 separator checkpoint、MAE checkpoint 和
manifest SHA-256。仅做分解时可设置 `reference_bank_required: false`，此时
reference bank 不参与推理；需要参考库匹配时才必须提供 reference bank 及其许可验收。

本仓库随 `services/xdecomposer_service/vendor/XDecomposer` 打包了 worker
所需的最小 XDecomposer MIT 源码副本，因此不再依赖开发机上的外部
`XDecomposer` 源码目录。真实权重仍然只通过本地 `models/` 挂载提供，不提交到 Git。

本地启动 XDecomposer worker：

```bash
CPICANN_XDECOMPOSER_BACKEND=remote \
CPICANN_XDECOMPOSER_SERVICE_URL=http://127.0.0.1:8100 \
PYTHONPATH=services/xdecomposer_service/src \
.venv/bin/uvicorn xdecomposer_service.main:app --host 127.0.0.1 --port 8100
```

另开一个终端启动 Web：

```bash
CPICANN_XDECOMPOSER_BACKEND=remote \
CPICANN_XDECOMPOSER_SERVICE_URL=http://127.0.0.1:8100 \
UV_CACHE_DIR=/tmp/cpicann-uv-cache .venv/bin/uv run streamlit run src/cpicann_xrd/web/app.py
```

## Docker 部署

CPU 镜像不包含真实权重。`models/` 和 `runs/` 通过卷挂载保留在宿主机。

只启动 Web app：

```bash
docker compose up -d --build
```

该命令只用于 CPICANN 单相识别；不会启动 XDecomposer worker，Web 中多相模式会保持禁用。

Web 地址：

```text
http://localhost:8501
```

内网服务器部署时，用服务器内网 IP 访问：

```text
http://<服务器内网IP>:8501
```

启动 Web + API：

```bash
docker compose --profile api up -d --build
```

该命令同样不会启动 XDecomposer worker。

API 地址：

```text
http://localhost:8000
```

完整启动 Web + API + XDecomposer worker：

```bash
docker compose \
  -f compose.yaml \
  -f compose.xdecomposer.yaml \
  --profile api \
  --profile xdecomposer \
  up -d --build
```

该命令会启动 `app`、`api` 和 `xdecomposer-worker`。Compose 挂载宿主机
`models/` 到容器内 `/app/models:ro`，并将 `models/xdecomposer` 挂载到
worker 的 `/app/models/xdecomposer:ro`，因此真实 XDecomposer manifest 应放在：

```text
models/xdecomposer/manifest.yaml
```

Dockerfile 会把 vendored XDecomposer 源码复制进 worker 镜像，但不会复制真实
checkpoint。

完整部署中，`app` 和 `api` 会等待 `xdecomposer-worker` 通过 `/readyz` 后再启动。
worker 启动时会预加载真实模型，避免第一次点击多相分解时才加载 checkpoint。

部署后检查：

```bash
docker compose -f compose.yaml -f compose.xdecomposer.yaml --profile api --profile xdecomposer ps
curl -fsS http://127.0.0.1:8501/
curl -fsS http://127.0.0.1:8000/healthz
curl -fsS http://127.0.0.1:8000/capabilities
curl -fsS http://127.0.0.1:8100/readyz
```

查看日志：

```bash
docker compose -f compose.yaml -f compose.xdecomposer.yaml --profile api --profile xdecomposer logs -f app api xdecomposer-worker
```

停止完整部署：

```bash
docker compose -f compose.yaml -f compose.xdecomposer.yaml --profile api --profile xdecomposer down
```

GPU worker overlay：

```bash
docker compose \
  -f compose.yaml \
  -f compose.xdecomposer.yaml \
  -f compose.gpu.yaml \
  --profile xdecomposer-gpu \
  up -d --build
```

HTTPS reverse-proxy examples are in `deploy/Caddyfile` and `deploy/nginx.conf`.

容器内真实模型 CLI 示例：

```bash
mkdir -p "$PWD/../cpicann-container-test"
chmod 777 "$PWD/../cpicann-container-test"

docker run --rm \
  -v "$PWD/models:/app/models:ro" \
  -v "$PWD/samples:/app/samples:ro" \
  -v "$PWD/../cpicann-container-test:/app/runs" \
  cpicann-xrd-app:local \
  cpicann-xrd batch \
  --backend cpicann \
  --input /app/samples/CPICANN识别/0-norm.txt \
  --input /app/samples/CPICANN识别/1-norm.txt \
  --input /app/samples/CPICANN识别/3-norm.txt \
  --include-must Zr O \
  --allowed-elements Li Zr O \
  --top-k 5 \
  --output /app/runs
```

容器使用非 root 用户运行，宿主机输出目录必须对容器用户可写。

## 开发验证

普通验证不需要真实权重：

```bash
UV_CACHE_DIR=/tmp/cpicann-uv-cache .venv/bin/uv run ruff format --check .
UV_CACHE_DIR=/tmp/cpicann-uv-cache .venv/bin/uv run ruff check .
UV_CACHE_DIR=/tmp/cpicann-uv-cache .venv/bin/uv run mypy src scripts
UV_CACHE_DIR=/tmp/cpicann-uv-cache .venv/bin/uv run pytest -q -m "not model"
```

真实模型验证需要本地权重：

```bash
UV_CACHE_DIR=/tmp/cpicann-uv-cache .venv/bin/uv run pytest -q -m model
```

构建包：

```bash
UV_CACHE_DIR=/tmp/cpicann-uv-cache .venv/bin/uv run python -m build
```

## 供应链与发布

- CPU Dockerfile 位于 `docker/Dockerfile.cpu`，默认基础镜像使用 GHCR。
- XDecomposer worker Dockerfile 位于 `docker/Dockerfile.xdecomposer`，并打包
  `services/xdecomposer_service/vendor/XDecomposer` 下的最小上游源码副本。
- GPU compose overlay 位于 `compose.gpu.yaml`。
- Caddy/Nginx 示例位于 `deploy/`。
- 运行时依赖清单位于 `docs/dependency-inventory.txt`，由 `uv export --frozen` 从 `uv.lock` 生成。
- 普通 CI 不访问真实权重 secret；真实模型 smoke test 在独立 workflow 中运行。
- 发布镜像不包含 `models/`、`runs/`、`.env` 或 checkpoint 文件。
- 发布前运行 `python scripts/release_preflight.py` 检查权重、token、运行数据和内部路径是否误入 Git。

## 参考文档

- 开发与验收计划：[DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)
- 发布候选验收：[docs/release_validation.md](docs/release_validation.md)
- 已知限制：[docs/known_limitations.md](docs/known_limitations.md)
- 安全边界：[SECURITY.md](SECURITY.md)
- XDecomposer 科学验证草稿：[docs/xdecomposer_scientific_validation.md](docs/xdecomposer_scientific_validation.md)
- 部署手册：[docs/deployment_runbook.md](docs/deployment_runbook.md)
- 回滚说明：[docs/rollback.md](docs/rollback.md)
- v0.2.0 发布说明：[docs/release_notes_v0.2.0.md](docs/release_notes_v0.2.0.md)
- 第三方许可说明：[docs/third_party_licenses.md](docs/third_party_licenses.md)
- 引用信息：[CITATION.cff](CITATION.cff)
- 第三方与权重再分发说明：[NOTICE](NOTICE)

## 当前限制

- 当前仓库不提交真实 CPICANN 权重。
- 模型权重、运行输出和 `.env` 文件不得提交到 Git。
- 预测置信度是模型排序分数，不是物相含量或定量组分。
- Web/API 适合可信内网或本机使用；公网部署应额外配置反向代理、HTTPS 和访问控制。
- XDecomposer 多相分解尚未完成科学验证，不应标记为生产已验证功能。
