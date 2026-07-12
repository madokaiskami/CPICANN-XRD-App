"""Minimal prediction service interface."""

from __future__ import annotations

import torch

from cpicann_xrd.exceptions import CpicannXrdError, ErrorCode
from cpicann_xrd.model.protocol import InferenceBackend
from cpicann_xrd.schemas import FilterSpec, PhaseRecord, PredictionItem, SamplePrediction


class PredictionService:
    """Small service layer wrapper around an inference backend."""

    def __init__(self, backend: InferenceBackend) -> None:
        self._backend = backend

    def predict_tensor(
        self,
        *,
        sample_id: str,
        source_filename: str,
        tensor: torch.Tensor,
        top_k: int = 5,
        filter_spec: FilterSpec | None = None,
    ) -> SamplePrediction:
        """Predict a single tensor using the configured backend."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        logits = self._backend.predict_logits(tensor)
        model_info = self._backend.model_info
        if logits.ndim != 2 or logits.shape[1] != model_info.num_classes:
            raise CpicannXrdError(
                ErrorCode.MODEL_OUTPUT_SIZE_MISMATCH,
                "模型输出类别数与模型信息不一致",
                details={
                    "expected": model_info.num_classes,
                    "actual_shape": list(logits.shape),
                },
            )
        if logits.shape[0] != 1:
            raise ValueError("Phase 2 minimal prediction service accepts one sample at a time")

        probabilities = torch.softmax(logits[0], dim=0)
        returned_top_k = min(top_k, model_info.num_classes)
        top_probabilities, top_indices = torch.topk(probabilities, k=returned_top_k)
        predictions = [
            PredictionItem(
                filtered_rank=rank,
                global_rank=rank,
                class_index=int(class_index.item()),
                phase=self._fake_phase_record(int(class_index.item())),
                raw_logit=float(logits[0, class_index].item()),
                unfiltered_probability=float(probability.item()),
                filtered_confidence=float(probability.item()),
            )
            for rank, (probability, class_index) in enumerate(
                zip(top_probabilities, top_indices, strict=True),
                start=1,
            )
        ]
        warnings = (
            ["FakeBackend result; not a real CPICANN prediction"]
            if model_info.backend == "fake"
            else []
        )
        return SamplePrediction(
            sample_id=sample_id,
            source_filename=source_filename,
            status="success",
            backend=model_info.backend,
            model_id=model_info.model_id,
            preprocessing_version=model_info.preprocessing_version,
            filter_spec=filter_spec or FilterSpec(),
            candidate_count_before_filter=model_info.num_classes,
            candidate_count_after_filter=model_info.num_classes,
            requested_top_k=top_k,
            returned_top_k=len(predictions),
            predictions=predictions,
            warnings=warnings,
        )

    @staticmethod
    def _fake_phase_record(class_index: int) -> PhaseRecord:
        return PhaseRecord(
            class_index=class_index,
            cod_id=f"FAKE-{class_index:05d}",
            formula=f"Fake{class_index}",
            reduced_formula=f"Fake{class_index}",
            elements=frozenset({"X"}),
            space_group=None,
            space_group_number=None,
        )
