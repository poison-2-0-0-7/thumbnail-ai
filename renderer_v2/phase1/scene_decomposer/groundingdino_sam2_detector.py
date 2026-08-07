"""GroundingDINO + SAM2.1 production implementation of Detector interface."""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
from loguru import logger
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from ultralytics import SAM

from .base import Detector
from ..schemas import Instance, InstanceClass
from ..config import Phase1Config, default_config
from ..model_registry import ModelRegistry


class GroundingDINOSAM2Detector(Detector):
    """Open-vocabulary detection via GroundingDINO and instance segmentation via SAM2.1."""

    def __init__(
        self,
        config: Phase1Config = default_config,
        registry: Optional[ModelRegistry] = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self._dino_model = None
        self._dino_processor = None
        self._sam2_model = None

    def _ensure_models_loaded(self) -> None:
        """Load GroundingDINO and SAM2.1 under ModelRegistry lifecycle management."""
        if self._dino_model is not None and self._sam2_model is not None:
            return

        def _loader() -> Tuple[AutoModelForZeroShotObjectDetection, AutoProcessor, SAM]:
            logger.info("Loading GroundingDINO model: {id}", id=self.config.grounding_dino_model_id)
            dino_proc = AutoProcessor.from_pretrained(
                self.config.grounding_dino_model_id,
                cache_dir=str(self.config.models_cache_dir),
            )
            dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.config.grounding_dino_model_id,
                cache_dir=str(self.config.models_cache_dir),
            ).to(device=self.config.device, dtype=torch.float32)
            dino_model.eval()

            logger.info("Loading SAM2.1 model: {id}", id=self.config.sam2_model_id)
            sam2_model = SAM(self.config.sam2_model_id)

            return (dino_model, dino_proc, sam2_model)

        if self.registry is not None:
            models = self.registry.load_model("groundingdino_sam2", _loader)
            self._dino_model, self._dino_processor, self._sam2_model = models
        else:
            self._dino_model, self._dino_processor, self._sam2_model = _loader()

    def detect(self, image: np.ndarray, class_prompts: List[str]) -> List[Instance]:
        """Detect and segment instances in image.

        Args:
            image: HxWx3 RGB uint8 image array.
            class_prompts: List of text class prompts.

        Returns:
            List of segmented Instance objects.
        """
        h, w, _ = image.shape
        self._ensure_models_loaded()
        instances: List[Instance] = []
        inst_idx = 0

        boxes_by_prompt = self._predict_boxes(image, class_prompts)

        for prompt, boxes in boxes_by_prompt.items():
            mapped_cls: InstanceClass = self._map_prompt_to_class(prompt)
            is_locked = prompt.lower() in [c.lower() for c in self.config.locked_classes]

            for bbox in boxes:
                xmin, ymin, xmax, ymax = bbox
                mask = self._segment_box(image, bbox)
                alpha_matte = mask.astype(np.float32)

                inst = Instance(
                    instance_id=f"{mapped_cls}_{inst_idx}",
                    cls=mapped_cls,
                    mask=mask,
                    alpha_matte=alpha_matte,
                    bbox=(int(xmin), int(ymin), int(xmax), int(ymax)),
                    depth_layer=0.5,
                    locked=is_locked,
                )
                instances.append(inst)
                inst_idx += 1

        return instances

    def _map_prompt_to_class(self, prompt: str) -> InstanceClass:
        p = prompt.lower().strip()
        if "person" in p or "creator" in p or "face" in p or "human" in p:
            return "creator"
        elif "logo" in p or "brand" in p or "icon" in p:
            return "logo"
        elif "product" in p or "phone" in p or "item" in p:
            return "product"
        else:
            return "other"

    def _predict_boxes(self, image: np.ndarray, prompts: List[str]) -> dict[str, List[tuple[int, int, int, int]]]:
        """Predict bounding boxes using GroundingDINO."""
        h, w, _ = image.shape
        results: dict[str, List[tuple[int, int, int, int]]] = {}

        pil_img = Image.fromarray(image)
        text_prompt = ". ".join(prompts) + "."

        inputs = self._dino_processor(images=pil_img, text=text_prompt, return_tensors="pt").to(
            device=self.config.device, dtype=torch.float32
        )
        with torch.no_grad():
            outputs = self._dino_model(**inputs)

        target_sizes = torch.tensor([[h, w]], device=self.config.device)
        results_dict = self._dino_processor.post_process_grounded_object_detection(
            outputs=outputs, input_ids=inputs.input_ids, target_sizes=target_sizes, threshold=0.25
        )[0]

        boxes = results_dict["boxes"].cpu().numpy()
        labels = results_dict["labels"]

        for prompt in prompts:
            prompt_boxes = []
            for box, label in zip(boxes, labels):
                if prompt.lower() in label.lower():
                    xmin, ymin, xmax, ymax = box
                    prompt_boxes.append((int(xmin), int(ymin), int(xmax), int(ymax)))
            results[prompt] = prompt_boxes

        return results

    def _segment_box(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        """Generate binary mask for bbox using SAM2.1."""
        h, w, _ = image.shape
        xmin, ymin, xmax, ymax = bbox
        box_np = [xmin, ymin, xmax, ymax]

        # SAM2 inference via Ultralytics API
        results = self._sam2_model.predict(image, bboxes=[box_np], verbose=False)
        if results and results[0].masks is not None:
            raw_mask = results[0].masks.data[0].cpu().numpy()
            return raw_mask > 0.5

        # Precise box mask
        mask = np.zeros((h, w), dtype=bool)
        x0, y0 = max(0, xmin), max(0, ymin)
        x1, y1 = min(w, xmax), min(h, ymax)
        mask[y0:y1, x0:x1] = True
        return mask
