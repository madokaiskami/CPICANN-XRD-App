"""Masked-logit confidence calculation."""

from __future__ import annotations

import torch

from cpicann_xrd.catalog.catalog import PhaseCatalog
from cpicann_xrd.catalog.element_filter import candidate_matches_filter
from cpicann_xrd.exceptions import CpicannXrdError, ErrorCode
from cpicann_xrd.schemas import FilterSpec, PredictionItem


def rank_predictions_from_logits(
    logits: torch.Tensor,
    *,
    catalog: PhaseCatalog,
    filter_spec: FilterSpec,
    top_k: int,
) -> tuple[list[PredictionItem], int, list[str]]:
    """Rank predictions using global logits and filtered masked softmax."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    one_dimensional_logits = _normalize_logits(logits)
    if one_dimensional_logits.shape[0] != len(catalog):
        raise CpicannXrdError(
            ErrorCode.MODEL_OUTPUT_SIZE_MISMATCH,
            "模型输出类别数与 catalog 记录数不一致",
            details={
                "logit_count": one_dimensional_logits.shape[0],
                "catalog_count": len(catalog),
            },
        )

    valid_records = [
        record for record in catalog.records if candidate_matches_filter(record, filter_spec)
    ]
    candidate_count_after_filter = len(valid_records)
    if candidate_count_after_filter == 0:
        raise CpicannXrdError(
            ErrorCode.NO_CANDIDATES_AFTER_FILTER,
            "元素过滤后没有可用候选相",
        )

    unfiltered_probabilities = torch.softmax(one_dimensional_logits, dim=0)
    global_order = torch.argsort(one_dimensional_logits, descending=True)
    global_ranks = {
        int(class_index.item()): rank
        for rank, class_index in enumerate(global_order, start=1)
    }

    valid_indices = torch.tensor(
        [record.class_index for record in valid_records],
        dtype=torch.long,
        device=one_dimensional_logits.device,
    )
    filtered_logits = one_dimensional_logits[valid_indices]
    filtered_probabilities = torch.softmax(filtered_logits, dim=0)
    filtered_order = torch.argsort(filtered_probabilities, descending=True)

    returned_top_k = min(top_k, candidate_count_after_filter)
    warnings = []
    if candidate_count_after_filter < top_k:
        warnings.append("候选数少于 Top-K，已返回全部过滤后候选。")

    predictions: list[PredictionItem] = []
    for filtered_rank, filtered_position in enumerate(
        filtered_order[:returned_top_k],
        start=1,
    ):
        valid_position = int(filtered_position.item())
        class_index = int(valid_indices[valid_position].item())
        predictions.append(
            PredictionItem(
                filtered_rank=filtered_rank,
                global_rank=global_ranks[class_index],
                class_index=class_index,
                phase=catalog.get(class_index),
                raw_logit=float(one_dimensional_logits[class_index].item()),
                unfiltered_probability=float(unfiltered_probabilities[class_index].item()),
                filtered_confidence=float(filtered_probabilities[valid_position].item()),
            )
        )
    return predictions, candidate_count_after_filter, warnings


def _normalize_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        return logits
    if logits.ndim == 2 and logits.shape[0] == 1:
        return logits[0]
    raise ValueError("logits must have shape (classes,) or (1, classes)")
