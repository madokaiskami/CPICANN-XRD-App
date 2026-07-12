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

## 当前限制

- Phase 0 只提供可安装包骨架和最小版本命令。
- 当前不包含真实 CPICANN 权重、类别目录或推理实现。
- 模型权重、运行输出和 `.env` 文件不得提交到 Git。
