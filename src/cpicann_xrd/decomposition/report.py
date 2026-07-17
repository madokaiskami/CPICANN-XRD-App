"""Chinese report rendering for XDecomposer decomposition results."""

from __future__ import annotations

from typing import Any

from cpicann_xrd.decomposition.schemas import XDecomposerRequest, XDecomposerResult

DISCLAIMER = (
    "本报告中的 component 是 XDecomposer 分解输出，不等同于已经确认的物相；"
    "estimated_weight 是基于分解谱强度的近似贡献值，不是 Rietveld 定量相含量。"
)


def render_decomposition_report(
    *,
    request: XDecomposerRequest,
    result: XDecomposerResult,
    metadata: dict[str, Any],
) -> str:
    """Render a Chinese Markdown report."""
    lines = [
        "# XDecomposer 多相分解报告",
        "",
        "## 运行摘要",
        "",
        f"- 样品 ID：`{request.sample_id}`",
        f"- 模型 ID：`{result.model_id}`",
        f"- 预处理协议：`{result.preprocessing_version}`",
        f"- 最大 component 数：{request.max_sources}",
        f"- Activity threshold：{request.activity_threshold}",
        f"- 重建 RMSE：{result.reconstruction_error:.8g}",
        "",
        "## Component 摘要",
        "",
        "| 显示序号 | 原始 slot | active probability | active | "
        "estimated weight | pattern SHA-256 |",
        "|---:|---:|---:|---|---:|---|",
    ]
    for display_index, component in enumerate(result.components, start=1):
        lines.append(
            "| "
            f"{display_index} | {component.original_slot_index} | "
            f"{component.active_probability:.6f} | "
            f"{'是' if component.is_active else '否'} | "
            f"{component.estimated_weight:.6f} | `{component.pattern_sha256}` |"
        )

    lines.extend(
        [
            "",
            "## 结果解释限制",
            "",
            DISCLAIMER,
            "",
            "activity probability 只表示该输出 slot 在模型判定下是否活跃；"
            "它不是 CPICANN 分类置信度，也不是相含量。低质量分解可能仍产生看似清晰的 component，"
            "必须结合原谱、重建谱、残差和后续专家复核。",
            "",
            "## 可复现信息",
            "",
            f"- 输入 SHA-256：`{metadata['input_sha256']}`",
            f"- Git commit：`{metadata['git_commit']}`",
            f"- Upstream commit：`{metadata['upstream_commit']}`",
            f"- Reference bank SHA-256：`{metadata['reference_bank_hash']}`",
            f"- Device：`{metadata['device']}`",
            "",
        ]
    )
    if result.warnings:
        lines.extend(["## 警告", ""])
        for warning in result.warnings:
            lines.append(f"- {warning}")
        lines.append("")
    return "\n".join(lines)
