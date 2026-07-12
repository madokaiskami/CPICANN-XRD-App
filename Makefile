.PHONY: setup format lint test test-model check build clean

setup:
	uv sync --all-extras

format:
	uv run ruff format .

lint:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy src

test:
	uv run pytest -q -m "not model"

test-model:
	uv run pytest -q -m model

check: lint test

build:
	uv run python -m build

clean:
	rm -rf build dist *.egg-info .coverage htmlcov .pytest_cache .mypy_cache .ruff_cache
