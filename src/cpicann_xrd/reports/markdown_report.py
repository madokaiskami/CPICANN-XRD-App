"""Chinese Markdown report exporter."""

from __future__ import annotations

from cpicann_xrd.schemas import DiagnosticRecord, FilterSpec, ModelInfo, SamplePrediction

DISCLAIMER = (
    "本结果为 CPICANN 单相分类模型在当前元素约束条件下生成的候选物相排序。\n"
    "过滤后置信度为候选集合内的相对条件分数，不代表实际物相含量，\n"
    "也不替代后续结构精修和实验专家确认。"
)


def render_summary_report(
    *,
    run_id: str,
    model_info: ModelInfo,
    preprocessing_version: str,
    filter_spec: FilterSpec,
    top_k: int,
    predictions: list[SamplePrediction],
    diagnostics: list[DiagnosticRecord],
    counts: dict[str, int],
) -> str:
    """Render a Chinese run summary report."""
    lines = [
        "# CPICANN-XRD 运行报告",
        "",
        "## 运行摘要",
        "",
        f"- Run ID：`{run_id}`",
        f"- 成功样品：{counts.get('success', 0)}",
        f"- 失败文件：{counts.get('failed', 0)}",
        f"- 忽略文件：{counts.get('ignored', 0)}",
        f"- Top-K：{top_k}",
        "",
        "## 模型和预处理信息",
        "",
        f"- 模型 ID：`{model_info.model_id}`",
        f"- 后端：`{model_info.backend}`",
        f"- 类别数：{model_info.num_classes}",
        f"- 预处理协议：`{preprocessing_version}`",
        f"- 权重 SHA-256：`{model_info.weight_sha256 or 'N/A'}`",
        f"- Catalog SHA-256：`{model_info.catalog_sha256 or 'N/A'}`",
        "",
        "## 元素过滤条件",
        "",
        f"- 必须包含元素：{_format_elements(filter_spec.include_must)}",
        f"- 允许元素范围：{_format_elements(filter_spec.allowed_elements)}",
        "",
        "## 每个样品 Top-K",
        "",
    ]
    for prediction in predictions:
        lines.extend(
            [
                f"### {prediction.sample_id}",
                "",
                "| 排名 | COD ID | 化学式 | 过滤后置信度 |",
                "|---:|---|---|---:|",
            ]
        )
        for item in prediction.predictions:
            lines.append(
                "| "
                f"{item.filtered_rank} | {item.phase.cod_id} | {item.phase.formula} | "
                f"{item.filtered_confidence:.6f} |"
            )
        if prediction.warnings:
            lines.append("")
            lines.append(f"警告：{'；'.join(prediction.warnings)}")
        lines.append("")

    lines.extend(["## 失败和忽略文件", ""])
    relevant_diagnostics = [
        diagnostic for diagnostic in diagnostics if diagnostic.status in {"failed", "ignored"}
    ]
    if relevant_diagnostics:
        lines.extend(["| 文件 | 状态 | 阶段 | 错误码 | 信息 |", "|---|---|---|---|---|"])
        for diagnostic in relevant_diagnostics:
            lines.append(
                "| "
                f"{diagnostic.source_filename} | {diagnostic.status} | {diagnostic.stage} | "
                f"{diagnostic.error_code.value if diagnostic.error_code else ''} | "
                f"{diagnostic.message} |"
            )
    else:
        lines.append("无失败或忽略文件。")

    lines.extend(
        [
            "",
            "## 结果解释限制",
            "",
            DISCLAIMER,
            "",
            "## 可复现信息",
            "",
            f"- Run ID：`{run_id}`",
            f"- 模型 ID：`{model_info.model_id}`",
            f"- Catalog SHA-256：`{model_info.catalog_sha256 or 'N/A'}`",
            f"- Top-K：{top_k}",
            "",
        ]
    )
    return "\n".join(lines)


def _format_elements(elements: frozenset[str] | None) -> str:
    if elements is None:
        return "未限制"
    if not elements:
        return "无"
    return "、".join(sorted(elements))
