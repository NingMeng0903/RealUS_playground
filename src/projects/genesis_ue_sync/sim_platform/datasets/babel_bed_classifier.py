"""Semantic bed-scene scoring for BABEL action text via HF zero-shot NLI (or fine-tuned head).

Keyword rules in the extractor are optional; this module is the preferred signal for *what the action is*.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

_HYP_TEMPLATE = "This example is {}."


def _maybe_stub_torchvision_for_text_transformers() -> None:
    """If torchvision is missing or incompatible with torch, register a minimal stub.

 ``transformers`` 5.x pulls vision helpers (e.g. Deformable DETR loss) while importing
    text models; a broken real ``torchvision`` then aborts BART load. This script only needs
    text NLI, so a stub is enough. Idempotent if called again.
    """
    import importlib
    import importlib.machinery
    import sys
    from types import ModuleType

    def _clear_torchvision_modules() -> None:
        for k in list(sys.modules.keys()):
            if k == "torchvision" or k.startswith("torchvision."):
                del sys.modules[k]

    try:
        importlib.import_module("torchvision")
        return
    except BaseException:
        _clear_torchvision_modules()

    def _pkg(name: str) -> ModuleType:
        mod = ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name=name, loader=None, is_package=True)
        mod.__path__ = []
        return mod

    def _mod(name: str) -> ModuleType:
        mod = ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name=name, loader=None, is_package=False)
        return mod

    functional = _mod("torchvision.transforms.v2.functional")
    v2 = _pkg("torchvision.transforms.v2")
    v2.functional = functional
    transforms = _pkg("torchvision.transforms")
    transforms.v2 = v2
    tv = _pkg("torchvision")
    tv.transforms = transforms
    sys.modules["torchvision"] = tv
    sys.modules["torchvision.transforms"] = transforms
    sys.modules["torchvision.transforms.v2"] = v2
    sys.modules["torchvision.transforms.v2.functional"] = functional


def _nli_entail_contrad_ids(config: Any) -> tuple[int, int]:
    entail_id, contrad_id = -1, -1
    mapping = getattr(config, "label2id", None) or {}
    for label, ind in mapping.items():
        low = str(label).lower()
        if low.startswith("entail"):
            entail_id = int(ind)
        elif low.startswith("contrad"):
            contrad_id = int(ind)
    if entail_id < 0:
        raise ValueError("NLI model config must define an entailment label in label2id")
    if contrad_id < 0:
        contrad_id = len(mapping) - 1 if entail_id == 0 else 0
    return entail_id, contrad_id


@dataclass
class BedClassifierScores:
    on_bed_likelihood: float
    other_surface_likelihood: float
    lying_or_sleeping_likelihood: float
    prone_likelihood: float
    sitting_likelihood: float
    standing_likelihood: float
    locomotion_likelihood: float
    raw: dict[str, Any]

    @property
    def bed_likelihood(self) -> float:
        return self.on_bed_likelihood

    @property
    def floor_or_ground_likelihood(self) -> float:
        return self.other_surface_likelihood

    @property
    def other_non_bed_likelihood(self) -> float:
        return self.other_surface_likelihood

    @property
    def positive_pose_likelihood(self) -> float:
        return max(self.lying_or_sleeping_likelihood, self.prone_likelihood, self.sitting_likelihood)


class ZeroShotBedClassifier:
    """Zero-shot NLI: natural-language scene hypotheses (multi-label scores).

    Uses AutoModelForSequenceClassification + tokenizer only (no ``pipeline()``).
    If ``torchvision`` is broken, a minimal stub is installed so ``transformers`` 5.x can import.
    """

    def __init__(
        self,
        *,
        model_name: str = "facebook/bart-large-mnli",
        device: int | str = -1,
    ) -> None:
        _maybe_stub_torchvision_for_text_transformers()
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.eval()
        if isinstance(device, int) and device >= 0:
            self._device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        elif str(device).lower() == "cuda":
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device("cpu")
        self._model.to(self._device)
        self._entail_id, self._contrad_id = _nli_entail_contrad_ids(self._model.config)
        self._model_name = str(getattr(self._model, "name_or_path", None) or model_name)
        self._on_bed_labels = [
            "The person is on a bed or mattress indoors.",
            "The person is using a bed, mattress, or bedding as the support surface.",
            "The body is supported by a bed rather than the floor, chair, or other furniture.",
        ]
        self._other_surface_labels = [
            "The person is on the floor or ground rather than on a bed.",
            "The person is sitting in a chair, stool, sofa, couch, or vehicle seat instead of on a bed.",
            "The body is supported by a non-bed surface such as the floor, chair, couch, or car seat.",
        ]
        self._lying_or_sleeping_labels = [
            "The person is lying down or sleeping.",
            "The person is resting flat, supine, asleep, or reclining.",
            "The body is in a low lying rest posture rather than upright.",
        ]
        self._prone_labels = [
            "The person is lying face down or prone.",
            "The body is belly-down against the support surface.",
        ]
        self._sitting_labels = [
            "The person is sitting with the torso upright.",
            "The body is in a seated posture rather than standing or walking.",
        ]
        self._standing_labels = [
            "The person is standing upright on their feet.",
            "The body is in an upright standing posture, not sitting or lying.",
        ]
        self._locomotion_labels = [
            "The person is walking jogging or running.",
            "The person is moving around the room mainly on foot.",
        ]

    def score(self, text: str) -> BedClassifierScores:
        text = (text or "").strip()
        if not text:
            return BedClassifierScores(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {"error": "empty_text"})

        premise = (
            "The following is a short English description of body movements in a motion capture clip. "
            f"Decide which situations apply: {text}"
        )
        import torch
        import torch.nn.functional as F
        from transformers.tokenization_utils_base import TruncationStrategy

        all_labels = (
            self._on_bed_labels
            + self._other_surface_labels
            + self._lying_or_sleeping_labels
            + self._prone_labels
            + self._sitting_labels
            + self._standing_labels
            + self._locomotion_labels
        )
        pairs = [[premise, _HYP_TEMPLATE.format(lab)] for lab in all_labels]
        inputs = self._tok(
            pairs,
            padding=True,
            truncation=TruncationStrategy.ONLY_FIRST,
            max_length=1024,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.inference_mode():
            logits = self._model(**inputs).logits
        pair_logits = logits[:, [self._contrad_id, self._entail_id]].float()
        probs = F.softmax(pair_logits, dim=-1)[:, 1]
        label_to_score = {lab: float(probs[i].item()) for i, lab in enumerate(all_labels)}

        def _mx(keys: list[str]) -> float:
            return max(float(label_to_score.get(lab, 0.0)) for lab in keys) if keys else 0.0

        on_bed_p = _mx(self._on_bed_labels)
        other_surface_p = _mx(self._other_surface_labels)
        lying_p = _mx(self._lying_or_sleeping_labels)
        prone_p = _mx(self._prone_labels)
        sitting_p = _mx(self._sitting_labels)
        standing_p = _mx(self._standing_labels)
        loco_p = _mx(self._locomotion_labels)
        sorted_idx = torch.argsort(probs, descending=True).tolist()
        labels_out = [all_labels[i] for i in sorted_idx]
        scores_out = [float(probs[i].item()) for i in sorted_idx]

        return BedClassifierScores(
            on_bed_likelihood=float(on_bed_p),
            other_surface_likelihood=float(other_surface_p),
            lying_or_sleeping_likelihood=float(lying_p),
            prone_likelihood=float(prone_p),
            sitting_likelihood=float(sitting_p),
            standing_likelihood=float(standing_p),
            locomotion_likelihood=float(loco_p),
            raw={
                "labels": labels_out,
                "scores": scores_out,
                "model": self._model_name,
                "multi_label": True,
                "groups": {
                    "on_bed_labels": list(self._on_bed_labels),
                    "other_surface_labels": list(self._other_surface_labels),
                    "lying_or_sleeping_labels": list(self._lying_or_sleeping_labels),
                    "prone_labels": list(self._prone_labels),
                    "sitting_labels": list(self._sitting_labels),
                    "standing_labels": list(self._standing_labels),
                    "locomotion_labels": list(self._locomotion_labels),
                },
            },
        )


class FinetunedBedClassifier:
    """Binary sequence classifier from a saved HF checkpoint (AutoModelForSequenceClassification)."""

    def __init__(self, model_dir: Path | str, *, device: str = "cpu") -> None:
        _maybe_stub_torchvision_for_text_transformers()
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_dir = Path(model_dir)
        self._tok = AutoTokenizer.from_pretrained(model_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self._model.eval()
        use_cuda = str(device).lower() == "cuda" and torch.cuda.is_available()
        self._device = torch.device("cuda" if use_cuda else "cpu")
        self._model.to(self._device)

    def score(self, text: str) -> BedClassifierScores:
        import torch

        text = (text or "").strip()
        if not text:
            return BedClassifierScores(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {"error": "empty_text"})
        enc = self._tok(text, return_tensors="pt", truncation=True, max_length=256)
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = self._model(**enc).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        n = int(probs.shape[0])
        if n == 2:
            bed_p, loco_p = float(probs[1]), float(probs[0])
        else:
            bed_p, loco_p = float(probs.max()), float(1.0 - probs.max())
        return BedClassifierScores(
            on_bed_likelihood=bed_p,
            other_surface_likelihood=0.0,
            lying_or_sleeping_likelihood=bed_p,
            prone_likelihood=0.0,
            sitting_likelihood=0.0,
            standing_likelihood=0.0,
            locomotion_likelihood=loco_p,
            raw={"probs": probs.tolist()},
        )


def build_classifier(
    mode: str,
    *,
    hf_model: str = "facebook/bart-large-mnli",
    finetuned_dir: Path | str | None = None,
    device: str = "cpu",
) -> ZeroShotBedClassifier | FinetunedBedClassifier:
    mode = mode.lower().strip()
    if mode == "zero_shot" or mode == "bert":
        dev: int | str = -1
        if device == "cuda":
            dev = 0
        return ZeroShotBedClassifier(model_name=hf_model, device=dev)
    if mode == "finetuned":
        if not finetuned_dir:
            raise ValueError("finetuned mode requires --finetuned-dir")
        return FinetunedBedClassifier(finetuned_dir, device=device)
    raise ValueError(f"Unknown classifier mode: {mode}")
