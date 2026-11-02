#!/usr/bin/env python3
"""Run one deterministic, GEO-like Laya inference as an installation check."""

from __future__ import annotations

import json
import os
import time

import laya
import torch
import transformers
from laya import Router


def main() -> None:
    device = os.environ.get("LAYA_DEVICE", "cuda")
    if device == "cuda":
        assert torch.cuda.is_available(), "CUDA requested but unavailable"
    state = (
        "GSM test sample. Organism: Homo sapiens. Source: liver biopsy "
        "from a healthy female donor."
    )
    questions = {
        "tissue": {
            "type": "choice",
            "instructions": "Which tissue best describes the biological sample?",
            "criteria": {
                "liver": "liver or hepatic tissue",
                "brain": "brain or neural tissue",
                "unknown": "not enough information",
            },
        },
        "sex_female": {
            "type": "noul",
            "instructions": "Is the donor female?",
        },
    }
    started = time.perf_counter()
    result = Router(device=device).predict(state, questions)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - started
    assert result["answers"]["tissue"]["choice"] == "liver", result
    payload = {
        "versions": {
            "laya": getattr(laya, "__version__", "unknown"),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "elapsed_seconds_including_lazy_load": elapsed_seconds,
        "result": result,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
