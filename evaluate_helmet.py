"""Avalia o checkpoint de capacete no split YOLO derivado do dataset de EPI.

O dataset ``construction-ppe-helmet`` contém os rótulos de uma única classe,
enquanto as imagens permanecem no dataset de origem ``construction-ppe``. Este
script reúne os dois diretórios e calcula as métricas COCO de detecção.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from rfdetr import RFDETRSmall

from model_hub import DEFAULT_MODEL_PATH, DEFAULT_MODEL_REPO, resolve_checkpoint


DEFAULT_CHECKPOINT = DEFAULT_MODEL_PATH
DEFAULT_LABELS_DIR = Path("construction-ppe-helmet/test/labels")
DEFAULT_IMAGES_DIR = Path("construction-ppe-helmet/test/images")
DEFAULT_REPORT = Path("outputs/evaluation/helmet_test_metrics.json")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcula métricas COCO para o detector de capacete no split de teste."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--hf-repo-id", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--prediction-threshold",
        type=float,
        default=0.001,
        help="Score mínimo mantido para o cálculo de AP (padrão: 0.001).",
    )
    return parser.parse_args()


def image_path_for(label_path: Path, images_dir: Path) -> Path:
    """Localiza a imagem de origem correspondente a um rótulo YOLO."""
    for suffix in IMAGE_SUFFIXES:
        candidate = images_dir / f"{label_path.stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Imagem para {label_path.name} não encontrada em {images_dir}")


def yolo_annotations(label_path: Path, image_id: int, width: int, height: int) -> list[dict]:
    """Converte caixas YOLO normalizadas em anotações COCO da classe helmet."""
    annotations: list[dict] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"Rótulo inválido em {label_path}: {line!r}")
        class_id, center_x, center_y, box_width, box_height = map(float, values)
        if int(class_id) != 0:
            continue
        box_width *= width
        box_height *= height
        x = center_x * width - box_width / 2
        y = center_y * height - box_height / 2
        annotations.append({
            "image_id": image_id,
            "category_id": 1,
            "bbox": [x, y, box_width, box_height],
            "area": box_width * box_height,
            "iscrowd": 0,
        })
    return annotations


def iou_xywh(first: list[float], second: list[float]) -> float:
    """Calcula IoU para caixas COCO no formato ``[x, y, largura, altura]``."""
    first_x2, first_y2 = first[0] + first[2], first[1] + first[3]
    second_x2, second_y2 = second[0] + second[2], second[1] + second[3]
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first_x2, second_x2)
    bottom = min(first_y2, second_y2)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = first[2] * first[3] + second[2] * second[3] - intersection
    return intersection / union if union else 0.0


def best_f1_at_iou_50(ground_truth: list[dict], detections: list[dict]) -> dict[str, float]:
    """Encontra o melhor ponto de F1 no sweep de score usando IoU de 0,50."""
    boxes_by_image: dict[int, list[list[float]]] = {}
    for annotation in ground_truth:
        boxes_by_image.setdefault(annotation["image_id"], []).append(annotation["bbox"])
    matched: dict[int, set[int]] = {image_id: set() for image_id in boxes_by_image}
    true_positives = 0
    false_positives = 0
    best = {"f1_50": 0.0, "precision_50": 0.0, "recall_50": 0.0, "confidence_at_best_f1": 0.0}
    total = len(ground_truth)
    for detection in sorted(detections, key=lambda item: item["score"], reverse=True):
        candidates = boxes_by_image.get(detection["image_id"], [])
        used = matched.setdefault(detection["image_id"], set())
        available = [index for index in range(len(candidates)) if index not in used]
        match = max(available, key=lambda index: iou_xywh(detection["bbox"], candidates[index]), default=None)
        if match is not None and iou_xywh(detection["bbox"], candidates[match]) >= 0.5:
            used.add(match)
            true_positives += 1
        else:
            false_positives += 1
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / total if total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best["f1_50"]:
            best = {
                "f1_50": f1,
                "precision_50": precision,
                "recall_50": recall,
                "confidence_at_best_f1": detection["score"],
            }
    return best


def main() -> None:
    args = parse_args()
    checkpoint = resolve_checkpoint(args.checkpoint, args.hf_repo_id)
    label_paths = sorted(args.labels_dir.glob("*.txt"))
    if not label_paths:
        raise FileNotFoundError(f"Nenhum rótulo YOLO encontrado em: {args.labels_dir}")

    dataset = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "helmet"}]}
    image_paths: list[Path] = []
    annotation_id = 1
    for image_id, label_path in enumerate(label_paths, start=1):
        image_path = image_path_for(label_path, args.images_dir)
        with Image.open(image_path) as image:
            width, height = image.size
        dataset["images"].append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})
        for annotation in yolo_annotations(label_path, image_id, width, height):
            annotation["id"] = annotation_id
            dataset["annotations"].append(annotation)
            annotation_id += 1
        image_paths.append(image_path)

    coco_ground_truth = COCO()
    coco_ground_truth.dataset = dataset
    coco_ground_truth.createIndex()
    model = RFDETRSmall.from_checkpoint(str(checkpoint))
    detections: list[dict] = []
    for image_id, image_path in enumerate(image_paths, start=1):
        image = np.asarray(Image.open(image_path).convert("RGB"))
        prediction = model.predict(image, threshold=args.prediction_threshold)
        boxes = np.asarray(getattr(prediction, "xyxy", []), dtype=float)
        scores = np.asarray(getattr(prediction, "confidence", []), dtype=float)
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            width, height = x2 - x1, y2 - y1
            if width <= 0 or height <= 0:
                continue
            detections.append({
                "image_id": image_id,
                "category_id": 1,
                "bbox": [float(x1), float(y1), float(width), float(height)],
                "score": float(score),
            })
        print(f"Imagem {image_id}/{len(image_paths)}: {len(boxes)} detecções", flush=True)

    if not detections:
        raise RuntimeError("O modelo não produziu detecções; não é possível calcular AP.")
    coco_detections = coco_ground_truth.loadRes(detections)
    evaluator = COCOeval(coco_ground_truth, coco_detections, "bbox")
    evaluator.params.imgIds = list(range(1, len(image_paths) + 1))
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    metrics = {
        "mAP_50_95": float(evaluator.stats[0]),
        "mAP_50": float(evaluator.stats[1]),
        "mAP_75": float(evaluator.stats[2]),
        "mAR_1": float(evaluator.stats[6]),
        "mAR_10": float(evaluator.stats[7]),
        "mAR_100": float(evaluator.stats[8]),
    }
    metrics.update(best_f1_at_iou_50(dataset["annotations"], detections))
    report = {
        "checkpoint": str(checkpoint),
        "labels_dir": str(args.labels_dir),
        "images_dir": str(args.images_dir),
        "images": len(image_paths),
        "ground_truth_helmets": len(dataset["annotations"]),
        "detections": len(detections),
        "prediction_threshold": args.prediction_threshold,
        "metrics": metrics,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Relatório salvo em: {args.report}")


if __name__ == "__main__":
    main()
