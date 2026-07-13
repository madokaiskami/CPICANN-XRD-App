# CPICANN-XRD-App

CPICANN-XRD-App is a reproducible XRD phase-identification application planned around the pretrained CPICANN single-phase model.

当前仓库处于早期分阶段开发。请以 [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) 作为需求和验收的唯一基线。

## 开发环境

推荐使用 Python 3.11 和 `uv`。如果本机使用 miniforge，可先创建项目虚拟环境：

```bash
mamba create -y -p .venv python=3.11 pip
source .venv/bin/activate
python -m pip install uv
uv sync --all-extras
```

如果系统已安装 `uv`，可直接运行：

```bash
uv sync --all-extras
```

## 基础命令

```bash
uv run cpicann-xrd --version
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest -q
uv run python -m build
```

## Docker 快速启动

CPU 镜像不包含真实 CPICANN 权重。`models/` 和 `runs/` 通过卷挂载保留在宿主机。

```bash
docker compose up --build
```

打开本地 Web 页面：

```text
http://localhost:8501
```

运行 CLI smoke test：

```bash
docker compose run --rm app cpicann-xrd doctor --backend fake
```

如果要在容器中使用真实模型，请先在宿主机按 manifest 放置权重到 `./models`，容器只读挂载该目录；公开镜像不会打包权重文件。

API 服务可通过 profile 启动：

```bash
docker compose --profile api up api
```

## 供应链

- CPU Dockerfile 位于 `docker/Dockerfile.cpu`，默认基础镜像使用 GHCR，避免本机依赖 Docker Hub。
- 运行时依赖清单位于 `docs/dependency-inventory.txt`，由 `uv export --frozen` 从 `uv.lock` 生成。
- 普通 CI 不访问真实权重 secret；真实模型 smoke test 在独立 workflow 中运行。
- 发布镜像不包含 `models/`、`runs/`、`.env` 或 checkpoint 文件。

## 当前限制

- 当前仓库不提交真实 CPICANN 权重。
- 模型权重、运行输出和 `.env` 文件不得提交到 Git。
