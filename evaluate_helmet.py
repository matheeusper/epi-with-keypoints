"""Avalia um checkpoint de uma classe diretamente em um split YOLO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from rfdetr import RFDETRSmall

from model_hub import DEFAULT_MODEL_PATH, DEFAULT_MODEL_REPO, resolve_checkpoint


DEFAULT_CHECKPOINT = DEFAULT_MODEL_PATH
DEFAULT_DATASET_DIR = Path("datasets/PPE_Detection")
DEFAULT_DATA_YAML = DEFAULT_DATASET_DIR / "data.yaml"
DEFAULT_LABELS_DIR = DEFAULT_DATASET_DIR / "test/labels"
DEFAULT_IMAGES_DIR = DEFAULT_DATASET_DIR / "test/images"
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
    parser.add_argument("--data-yaml", type=Path, default=DEFAULT_DATA_YAML)
    parser.add_argument(
        "--class",
        dest="target_class",
        default="helmet",
        help="Nome ou ID original da classe no data.yaml (padrão: helmet).",
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--resolution",
        type=int,
        help="Força a resolução ao carregar um checkpoint sem model_config.",
    )
    parser.add_argument(
        "--prediction-threshold",
        type=float,
        default=0.001,
        help="Score mínimo mantido para o cálculo de AP (padrão: 0.001).",
    )
    return parser.parse_args()


def load_class_names(data_yaml: Path) -> dict[int, str]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    raw_names = data.get("names")
    if isinstance(raw_names, list):
        return {index: str(name) for index, name in enumerate(raw_names)}
    if isinstance(raw_names, dict):
        return {int(class_id): str(name) for class_id, name in raw_names.items()}
    raise ValueError(f"Campo 'names' inválido em {data_yaml}.")


def resolve_class(target: str, names: dict[int, str]) -> tuple[int, str]:
    if target.isdigit():
        class_id = int(target)
        if class_id not in names:
            raise ValueError(f"ID de classe inexistente: {class_id}.")
        return class_id, names[class_id]
    for class_id, name in names.items():
        if name.casefold() == target.casefold():
            return class_id, name
    available = ", ".join(f"{class_id}:{name}" for class_id, name in sorted(names.items()))
    raise ValueError(f"Classe desconhecida: {target!r}. Disponíveis: {available}")


def yolo_annotations(
    label_path: Path, image_id: int, width: int, height: int, source_class_id: int
) -> list[dict]:
    """Converte caixas ou polígonos YOLO da classe escolhida para caixas COCO."""
    annotations: list[dict] = []
    if not label_path.is_file():
        return annotations
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 5:
            raise ValueError(f"Rótulo inválido em {label_path}: {line!r}")
        try:
            class_id = int(values[0])
            coordinates = [float(value) for value in values[1:]]
        except ValueError as error:
            raise ValueError(f"Rótulo inválido em {label_path}: {line!r}") from error
        if class_id != source_class_id:
            continue
        if len(coordinates) == 4:
            center_x, center_y, box_width, box_height = coordinates
            x = (center_x - box_width / 2) * width
            y = (center_y - box_height / 2) * height
            box_width *= width
            box_height *= height
        elif len(coordinates) >= 6 and len(coordinates) % 2 == 0:
            xs, ys = coordinates[0::2], coordinates[1::2]
            x_min, x_max = min(xs) * width, max(xs) * width
            y_min, y_max = min(ys) * height, max(ys) * height
            x, y = x_min, y_min
            box_width, box_height = x_max - x_min, y_max - y_min
        else:
            raise ValueError(f"Rótulo inválido em {label_path}: {line!r}")
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
    if args.resolution is not None and args.resolution <= 0:
        raise ValueError("--resolution deve ser maior que zero.")
    checkpoint = resolve_checkpoint(args.checkpoint, args.hf_repo_id)
    names = load_class_names(args.data_yaml)
    source_class_id, class_name = resolve_class(args.target_class, names)
    image_paths = sorted(
        path for path in args.images_dir.iterdir()
        if path.is_file() and path.suffix in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em: {args.images_dir}")

    dataset = {"images": [], "annotations": [], "categories": [{"id": 1, "name": class_name}]}
    annotation_id = 1
    for image_id, image_path in enumerate(image_paths, start=1):
        label_path = args.labels_dir / f"{image_path.stem}.txt"
        with Image.open(image_path) as image:
            width, height = image.size
        dataset["images"].append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})
        for annotation in yolo_annotations(
            label_path, image_id, width, height, source_class_id
        ):
            annotation["id"] = annotation_id
            dataset["annotations"].append(annotation)
            annotation_id += 1
    if not dataset["annotations"]:
        raise ValueError(f"Nenhuma anotação da classe {class_name!r} encontrada no teste.")

    coco_ground_truth = COCO()
    coco_ground_truth.dataset = dataset
    coco_ground_truth.createIndex()
    load_args = {"resolution": args.resolution} if args.resolution is not None else {}
    model = RFDETRSmall.from_checkpoint(str(checkpoint), **load_args)
    evaluation_resolution = int(model.model_config.resolution)
    print(
        f"Avaliando {class_name!r} (ID original {source_class_id}) em "
        f"{len(image_paths)} imagens, resolução {evaluation_resolution}px."
    )
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
        "class_name": class_name,
        "source_class_id": source_class_id,
        "data_yaml": str(args.data_yaml),
        "labels_dir": str(args.labels_dir),
        "images_dir": str(args.images_dir),
        "images": len(image_paths),
        "ground_truth_objects": len(dataset["annotations"]),
        "detections": len(detections),
        "prediction_threshold": args.prediction_threshold,
        "evaluation_resolution": evaluation_resolution,
        "metrics": metrics,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Relatório salvo em: {args.report}")


if __name__ == "__main__":
    main()
