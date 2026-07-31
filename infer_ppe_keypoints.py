"""Valida o uso de capacete combinando detecção de EPI e keypoints corporais.

O modelo treinado detecta os capacetes. O modelo pré-treinado de keypoints detecta
as pessoas e localiza cabeça, braços e pernas. A união dos resultados permite
atribuir um capacete à pessoa correta e verificar se ele está na região da cabeça.

Exemplos:
    uv run python infer_ppe_keypoints.py --image caminho/para/imagem.jpg
    uv run python infer_ppe_keypoints.py --video caminho/para/video.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rfdetr import RFDETRKeypointPreview, RFDETRSmall


# Caminho padrão para o melhor checkpoint obtido no treinamento de EPIs.
DEFAULT_PPE_CHECKPOINT = Path("models/helmet_only_576_best.pth")
DEFAULT_OUTPUT_DIR = Path("outputs")
# Traduções aceitas entre os nomes de classes em português e inglês.
PPE_ALIASES = {
    "capacete": "capacete", "helmet": "capacete", "colete": "colete", "vest": "colete",
    "luva": "luvas", "luvas": "luvas", "glove": "luvas", "gloves": "luvas",
    "bota": "botas", "botas": "botas", "boot": "botas", "boots": "botas",
}
# A regra atual de conformidade avalia exclusivamente o uso de capacete.
REQUIRED_PPE = ("capacete",)
# Classes que existem no dataset, mas não participam da verificação de capacete.
IGNORED_DETECTION_CLASSES = {"person", "pessoa", "none", "nenhum", "background", "__background__"}
# Índices dos keypoints do padrão COCO. Os demais ficam preparados para futuras regras.
COCO_HEAD = (0, 1, 2, 3, 4)
COCO_WRISTS = (9, 10)
COCO_ANKLES = (15, 16)


def parse_args() -> argparse.Namespace:
    """Lê os parâmetros informados na linha de comando."""
    parser = argparse.ArgumentParser(
        description="Detecta EPIs e keypoints de pessoas em imagens ou vídeos."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=Path, help="Imagem de entrada.")
    input_group.add_argument("--video", type=Path, help="Vídeo de entrada.")
    parser.add_argument(
        "--ppe-checkpoint",
        type=Path,
        default=DEFAULT_PPE_CHECKPOINT,
        help=f"Checkpoint treinado de EPIs (padrão: {DEFAULT_PPE_CHECKPOINT}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Arquivo de saída. Padrão: outputs/<entrada>/annotated/<entrada>_ppe_keypoints.*.",
    )
    parser.add_argument("--report", type=Path, default=None, help="Relatório JSON de conformidade.")
    parser.add_argument("--ppe-threshold", type=float, default=0.35, help="Limiar do detector de capacete.")
    parser.add_argument("--keypoint-threshold", type=float, default=0.55, help="Limiar do detector de pessoas.")
    parser.add_argument(
        "--person-nms-iou",
        type=float,
        default=0.75,
        help="IoU máximo entre caixas de pessoa antes de suprimir duplicatas; padrão 0.75.",
    )
    parser.add_argument(
        "--keypoint-confidence",
        type=float,
        default=0.30,
        help="Confiança mínima para desenhar cada keypoint.",
    )
    parser.add_argument(
        "--hide-person-boxes",
        action="store_true",
        help="Oculta as caixas e rótulos de pessoa; mantém apenas os keypoints.",
    )
    parser.add_argument(
        "--draw-keypoints",
        action="store_true",
        help="Desenha os keypoints; por padrão eles ficam ocultos para uma imagem minimalista.",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="Codec de quatro caracteres do vídeo de saída (padrão: mp4v).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Processa no máximo esta quantidade de quadros (útil para testes).",
    )
    args = parser.parse_args()
    if len(args.codec) != 4:
        parser.error("--codec deve ter exatamente quatro caracteres, por exemplo mp4v.")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames deve ser maior que zero.")
    return args


def class_name(detections: object, index: int, fallback: str) -> str:
    """Obtém o nome da classe preservado pelo checkpoint, quando disponível."""
    data = getattr(detections, "data", {})
    names = data.get("class_name") if hasattr(data, "get") else None
    if names is not None and len(names) > index:
        return str(names[index])
    return fallback


def overlaps(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> bool:
    """Retorna se dois retângulos de texto ocupam a mesma área."""
    return not (
        first[2] <= second[0]
        or second[2] <= first[0]
        or first[3] <= second[1]
        or second[3] <= first[1]
    )


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    color: str,
    occupied_labels: list[tuple[int, int, int, int]],
    image_size: tuple[int, int],
) -> None:
    """Desenha uma etiqueta próxima à caixa, sem encobrir outra etiqueta."""
    font = ImageFont.load_default()
    width, height = image_size
    text_box = draw.textbbox((0, 0), text, font=font)
    label_width = text_box[2] - text_box[0] + 4
    label_height = text_box[3] - text_box[1] + 4
    x, y = map(int, xy)

    # Prioriza posições próximas à caixa e desloca verticalmente quando necessário.
    candidates = [
        (x, y + offset)
        for offset in (0, -label_height, label_height, -2 * label_height, 2 * label_height)
    ]
    candidates += [(x + label_width, y), (x - label_width, y)]
    for candidate_x, candidate_y in candidates:
        candidate_x = min(max(0, candidate_x), max(0, width - label_width))
        candidate_y = min(max(0, candidate_y), max(0, height - label_height))
        rectangle = (
            candidate_x,
            candidate_y,
            candidate_x + label_width,
            candidate_y + label_height,
        )
        if not any(overlaps(rectangle, other) for other in occupied_labels):
            break
    else:
        # Em imagens muito carregadas, preserva a legibilidade mesmo sem posição ideal.
        candidate_x = min(max(0, x), max(0, width - label_width))
        candidate_y = min(max(0, y), max(0, height - label_height))
        rectangle = (
            candidate_x,
            candidate_y,
            candidate_x + label_width,
            candidate_y + label_height,
        )

    draw.rectangle(rectangle, fill=color)
    draw.text((candidate_x + 2, candidate_y + 2), text, fill="white", font=font)
    occupied_labels.append(rectangle)


def format_confidence(value: float) -> str:
    """Formata probabilidades e evita apresentar scores brutos como porcentagem."""
    if 0.0 <= value <= 1.0:
        return f"{value:.0%}"
    return f"score {value:.2f}"


def ppe_kind(name: str) -> str | None:
    """Normaliza o nome de uma classe para o nome interno do EPI."""
    normalized = name.lower().replace("_", " ").replace("-", " ").strip()
    if normalized.startswith("no "):
        return None
    return PPE_ALIASES.get(normalized)


def missing_ppe_kind(name: str) -> str | None:
    """Identifica classes negativas, como ``no_helmet`` (reservado para uso futuro)."""
    normalized = name.lower().replace("_", " ").replace("-", " ").strip()
    return PPE_ALIASES.get(normalized[3:]) if normalized.startswith("no ") else None


def is_ignored_class(name: str) -> bool:
    """Informa se a classe não deve aparecer nem participar da validação."""
    return name.lower().strip() in IGNORED_DETECTION_CLASSES


def is_target_ppe(name: str) -> bool:
    """A versão atual da pipeline avalia somente capacetes positivos."""
    return ppe_kind(name) == "capacete" and missing_ppe_kind(name) is None


def point_distance(point: np.ndarray, candidates: np.ndarray) -> float:
    """Calcula a menor distância euclidiana entre um ponto e vários keypoints."""
    return float(np.linalg.norm(candidates - point, axis=1).min())


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    """Calcula a interseção sobre união de duas caixas no formato ``xyxy``."""
    left_top = np.maximum(first[:2], second[:2])
    right_bottom = np.minimum(first[2:], second[2:])
    intersection_size = np.maximum(0, right_bottom - left_top)
    intersection = float(intersection_size[0] * intersection_size[1])
    first_area = float(np.prod(np.maximum(0, first[2:] - first[:2])))
    second_area = float(np.prod(np.maximum(0, second[2:] - second[:2])))
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def suppress_duplicate_people(keypoints: object, max_iou: float) -> np.ndarray:
    """Mantém uma caixa por pessoa e remove detecções de pose quase idênticas."""
    boxes = np.asarray(getattr(keypoints, "data", {}).get("xyxy", []), dtype=float)
    if not len(boxes):
        return np.array([], dtype=int)
    scores = np.asarray(getattr(keypoints, "detection_confidence", np.ones(len(boxes))))
    # O valor pode ser um score bruto, mas ainda é adequado para ordenar as duplicatas.
    order = np.argsort(scores)[::-1]
    kept: list[int] = []
    for index in order:
        if all(box_iou(boxes[index], boxes[kept_index]) < max_iou for kept_index in kept):
            kept.append(int(index))
    return np.array(sorted(kept), dtype=int)


def validate_position(kind: str, center: np.ndarray, person: dict) -> str:
    """Valida a região do EPI com keypoints COCO; nunca infere ausência por oclusão."""
    # A altura da pessoa normaliza as distâncias e evita regras fixas em pixels.
    x1, y1, x2, y2 = person["box"]
    height = max(y2 - y1, 1.0)
    points, visible = person["points"], person["visible"]
    if kind == "capacete":
        candidates = points[list(COCO_HEAD)][visible[list(COCO_HEAD)]]
        if len(candidates):
            return "validado" if point_distance(center, candidates) <= height * 0.22 else "fora_da_regiao"
        return "nao_verificavel"
    if kind == "luvas":
        candidates = points[list(COCO_WRISTS)][visible[list(COCO_WRISTS)]]
        if len(candidates):
            return "validado" if point_distance(center, candidates) <= height * 0.28 else "fora_da_regiao"
        return "nao_verificavel"
    if kind == "botas":
        candidates = points[list(COCO_ANKLES)][visible[list(COCO_ANKLES)]]
        if len(candidates):
            return "validado" if point_distance(center, candidates) <= height * 0.28 else "fora_da_regiao"
        return "nao_verificavel"
    # Colete: faixa central do tronco. Não depende de articulações possivelmente ocultas.
    return "validado" if x1 <= center[0] <= x2 and y1 + height * .18 <= center[1] <= y1 + height * .78 else "fora_da_regiao"


def helmet_person_distance(center: np.ndarray, person: dict) -> float:
    """Mede quão compatível é um capacete com uma pessoa específica.

    Quando há keypoints confiáveis da cabeça, a associação usa essa região. Isso
    evita atribuir os dois capacetes à mesma pessoa quando caixas corporais de
    pessoas próximas se sobrepõem. A caixa corporal é usada apenas como fallback.
    """
    x1, y1, x2, y2 = person["box"]
    height = max(y2 - y1, 1.0)
    head_points = person["points"][list(COCO_HEAD)][person["visible"][list(COCO_HEAD)]]
    if len(head_points):
        return point_distance(center, head_points) / height

    # Sem uma cabeça confiável, preserva o comportamento anterior com peso menor.
    body_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    inside_box = x1 <= center[0] <= x2 and y1 <= center[1] <= y2
    return 1.0 + (0.0 if inside_box else float(np.linalg.norm(center - body_center)) / height)


def fuse_detections(
    ppe_detections: object,
    keypoints: object,
    min_kp_confidence: float,
    person_indices: np.ndarray,
) -> tuple[list[dict], dict[int, dict]]:
    """Associa EPIs à caixa de pessoa mais próxima e valida a geometria corporal."""
    # Cada pessoa possui uma caixa corporal e uma lista de keypoints prevista pelo modelo de pose.
    boxes = getattr(keypoints, "data", {}).get("xyxy")
    if boxes is None:
        return [], {}
    all_points = np.asarray(getattr(keypoints, "xy"))
    all_confidences = np.asarray(getattr(keypoints, "keypoint_confidence"))
    all_visibility = np.asarray(
        getattr(keypoints, "visible", np.ones(all_confidences.shape, dtype=bool))
    )
    boxes = np.asarray(boxes)[person_indices]
    points = all_points[person_indices]
    confidences = all_confidences[person_indices]
    visibility = all_visibility[person_indices]
    # Monta a estrutura que será escrita no relatório JSON ao final da execução.
    people = [
        {"id": index + 1, "box": np.asarray(box, dtype=float), "points": points[index],
         "visible": visibility[index] & (confidences[index] >= min_kp_confidence),
         "ppe": {kind: {"status": "ausente", "detections": []} for kind in REQUIRED_PPE}}
        for index, box in enumerate(np.asarray(boxes))
    ]
    validation: dict[int, dict] = {}
    for index, box in enumerate(np.asarray(getattr(ppe_detections, "xyxy"))):
        name = class_name(ppe_detections, index, "desconhecido")
        # Mantém apenas a classe positiva de capacete; todas as demais são descartadas.
        if is_ignored_class(name) or not is_target_ppe(name):
            continue
        kind = ppe_kind(name)
        missing_kind = None
        if kind is None or not people:
            validation[index] = {"status": "nao_associado", "person_id": None, "kind": kind}
            continue
        # O centro da caixa do capacete decide a qual pessoa ele pertence.
        center = (box[:2] + box[2:]) / 2
        # Associa pelo keypoint de cabeça mais próximo; a caixa corporal é fallback.
        person = min(people, key=lambda candidate: helmet_person_distance(center, candidate))
        status = "ausente_detectado" if missing_kind else validate_position(kind, center, person)
        item = {"status": status, "detection_index": index, "class_name": name}
        person["ppe"][kind]["detections"].append(item)
        if status in ("validado", "nao_verificavel", "ausente_detectado"):
            person["ppe"][kind]["status"] = status
        validation[index] = {"status": status, "person_id": person["id"], "kind": kind}
    # Resume quantos EPIs obrigatórios foram geometricamente validados por pessoa.
    for person in people:
        person["conformes"] = sum(v["status"] == "validado" for v in person["ppe"].values())
    return people, validation


def draw_ppe(
    draw: ImageDraw.ImageDraw,
    detections: object,
    occupied_labels: list[tuple[int, int, int, int]],
    image_size: tuple[int, int],
    validation: dict[int, dict],
) -> int:
    """Desenha somente os capacetes que passaram pelo filtro da pipeline."""
    boxes = np.asarray(getattr(detections, "xyxy"))
    confidences = np.asarray(getattr(detections, "confidence"))
    class_ids = np.asarray(getattr(detections, "class_id"))

    drawn_count = 0
    for index, (box, confidence, class_id) in enumerate(zip(boxes, confidences, class_ids)):
        name = class_name(detections, index, f"EPI {int(class_id)}")
        if is_ignored_class(name) or not is_target_ppe(name):
            continue
        x1, y1, x2, y2 = box.astype(float)
        # Verde: posição compatível com a cabeça; laranja: não foi possível validar; vermelho: alerta.
        result = validation.get(index, {})
        status = result.get("status", "nao_associado")
        color = "#34c759" if status == "validado" else "#ff9500" if status == "nao_verificavel" else "#ff3b30"
        marker = "OK" if status == "validado" else "?" if status == "nao_verificavel" else "X"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        draw_label(
            draw,
            (x1, y1 - 16),
            f"{name} {marker}",
            color,
            occupied_labels,
            image_size,
        )
        drawn_count += 1
    return drawn_count


def draw_keypoints(
    draw: ImageDraw.ImageDraw,
    keypoints: object,
    min_confidence: float,
    draw_person_boxes: bool,
    occupied_labels: list[tuple[int, int, int, int]],
    image_size: tuple[int, int],
    people: list[dict],
    draw_points: bool,
    person_indices: np.ndarray,
) -> int:
    """Desenha caixas de alerta das pessoas e, opcionalmente, seus keypoints."""
    all_points = np.asarray(getattr(keypoints, "xy"))
    all_point_confidences = np.asarray(getattr(keypoints, "keypoint_confidence"))
    all_visible = np.asarray(
        getattr(keypoints, "visible", np.ones(all_point_confidences.shape, dtype=bool))
    )
    points_per_person = all_points[person_indices]
    point_confidences = all_point_confidences[person_indices]
    person_confidences = np.asarray(getattr(keypoints, "detection_confidence"))[person_indices]
    visible = all_visible[person_indices]

    for person_index, (points, confidences, is_visible, person_confidence) in enumerate(
        zip(points_per_person, point_confidences, visible, person_confidences), start=1
    ):
        # Pontos de baixa confiança não são desenhados na visualização opcional.
        valid = (confidences >= min_confidence) & is_visible
        if draw_points:
            for (x, y), is_valid in zip(points, valid):
                if is_valid:
                    radius = 3
                    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="#007aff")

        # A caixa azul identifica a pessoa; as caixas vermelhas identificam os EPIs.
        boxes = np.asarray(getattr(keypoints, "data", {}).get("xyxy", []))[person_indices]
        if draw_person_boxes and boxes is not None and len(boxes) >= person_index:
            x1, y1, x2, y2 = np.asarray(boxes[person_index - 1]).astype(float)
            # A pessoa fica verde apenas quando há capacete na região esperada da cabeça.
            helmet_status = people[person_index - 1]["ppe"]["capacete"]["status"]
            is_protected = helmet_status == "validado"
            color = "#34c759" if is_protected else "#ff3b30"
            # Usa somente caracteres ASCII, pois a fonte padrão do Pillow não
            # possui cobertura garantida para símbolos como "✓".
            label = f"P{person_index} OK" if is_protected else f"P{person_index} ALERTA"
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            draw_label(
                draw,
                (x1, max(0, y1 - 14)),
                label,
                color,
                occupied_labels,
                image_size,
            )
    return len(points_per_person)


def annotate_frame(
    image_array: np.ndarray,
    ppe_model: RFDETRSmall,
    keypoint_model: RFDETRKeypointPreview,
    args: argparse.Namespace,
) -> tuple[Image.Image, int, int, dict]:
    """Executa a pipeline completa em um frame RGB e retorna a imagem e o relatório."""
    ppe_detections = ppe_model.predict(image_array, threshold=args.ppe_threshold)
    people_keypoints = keypoint_model.predict(image_array, threshold=args.keypoint_threshold)

    annotated = Image.fromarray(image_array).convert("RGB")
    draw = ImageDraw.Draw(annotated)
    occupied_labels: list[tuple[int, int, int, int]] = []
    person_indices = suppress_duplicate_people(people_keypoints, args.person_nms_iou)
    people, validation = fuse_detections(
        ppe_detections, people_keypoints, args.keypoint_confidence, person_indices
    )
    ppe_count = draw_ppe(
        draw, ppe_detections, occupied_labels, annotated.size, validation
    )
    people_count = draw_keypoints(
        draw,
        people_keypoints,
        args.keypoint_confidence,
        not args.hide_person_boxes,
        occupied_labels,
        annotated.size,
        people,
        args.draw_keypoints,
        person_indices,
    )
    report = {
        "pessoas": [
            {"id": person["id"], "epis": person["ppe"], "epis_validados": person["conformes"]}
            for person in people
        ],
        "epis_nao_associados": [
            index for index, result in validation.items() if result["person_id"] is None
        ],
    }
    return annotated, ppe_count, people_count, report


def load_models(checkpoint: Path) -> tuple[RFDETRSmall, RFDETRKeypointPreview]:
    """Carrega os modelos somente uma vez, inclusive durante inferência em vídeo."""
    return RFDETRSmall.from_checkpoint(str(checkpoint)), RFDETRKeypointPreview()


def save_report(path: Path, report: dict) -> None:
    """Grava o relatório JSON garantindo que o diretório exista."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def default_output_paths(input_path: Path, extension: str) -> tuple[Path, Path]:
    """Organiza cada inferência em mídia anotada e relatório separados.

    Exemplo para ``videos/obra.mp4``:
    ``outputs/obra/annotated/obra_ppe_keypoints.mp4`` e
    ``outputs/obra/reports/obra_ppe_keypoints.json``.
    """
    result_dir = DEFAULT_OUTPUT_DIR / input_path.stem
    filename = f"{input_path.stem}_ppe_keypoints"
    return (
        result_dir / "annotated" / f"{filename}{extension}",
        result_dir / "reports" / f"{filename}.json",
    )


def infer_image(args: argparse.Namespace, ppe_model: RFDETRSmall, keypoint_model: RFDETRKeypointPreview) -> None:
    """Processa uma única imagem e preserva o formato de saída original."""
    assert args.image is not None
    default_output_path, default_report_path = default_output_paths(args.image, ".jpg")
    output_path = args.output or default_output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_array = np.asarray(Image.open(args.image).convert("RGB"))
    annotated, ppe_count, people_count, report = annotate_frame(
        image_array, ppe_model, keypoint_model, args
    )
    annotated.save(output_path)

    report_path = args.report or (output_path.with_suffix(".json") if args.output else default_report_path)
    report["image"] = str(args.image)
    save_report(report_path, report)

    print(f"EPIs detectados: {ppe_count}")
    print(f"Pessoas com keypoints: {people_count}")
    print(f"Imagem anotada salva em: {output_path}")
    print(f"Relatório de conformidade salvo em: {report_path}")


def infer_video(args: argparse.Namespace, ppe_model: RFDETRSmall, keypoint_model: RFDETRKeypointPreview) -> None:
    """Processa todos os quadros de um vídeo e escreve uma cópia anotada em MP4."""
    assert args.video is not None
    default_output_path, default_report_path = default_output_paths(args.video, ".mp4")
    output_path = args.output or default_output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {args.video}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if np.isfinite(fps) and fps > 0 else 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"O vídeo não possui dimensões válidas: {args.video}")
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*args.codec), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            f"Não foi possível criar o vídeo de saída: {output_path}. "
            "Tente --codec avc1 ou outro codec disponível no sistema."
        )

    frames: list[dict] = []
    frame_index = 0
    total_ppe = 0
    total_people = 0
    try:
        while args.max_frames is None or frame_index < args.max_frames:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            annotated, ppe_count, people_count, frame_report = annotate_frame(
                frame_rgb, ppe_model, keypoint_model, args
            )
            writer.write(cv2.cvtColor(np.asarray(annotated), cv2.COLOR_RGB2BGR))
            frame_report.update({"quadro": frame_index, "tempo_segundos": frame_index / fps})
            frames.append(frame_report)
            total_ppe += ppe_count
            total_people += people_count
            frame_index += 1
            print(f"Quadro {frame_index}: {ppe_count} EPIs, {people_count} pessoas", flush=True)
    finally:
        capture.release()
        writer.release()

    if not frame_index:
        raise RuntimeError(f"Nenhum quadro pôde ser lido do vídeo: {args.video}")
    report_path = args.report or (output_path.with_suffix(".json") if args.output else default_report_path)
    save_report(report_path, {
        "video": str(args.video),
        "fps": fps,
        "quadros_processados": frame_index,
        "epis_detectados_total": total_ppe,
        "pessoas_detectadas_total": total_people,
        "quadros": frames,
    })
    print(f"Quadros processados: {frame_index}")
    print(f"Vídeo anotado salvo em: {output_path}")
    print(f"Relatório de conformidade salvo em: {report_path}")


def main() -> None:
    """Carrega modelos e encaminha a inferência para imagem ou vídeo."""
    args = parse_args()
    input_path = args.image or args.video
    if input_path is None or not input_path.is_file():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {input_path}")
    if not args.ppe_checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint de EPIs não encontrado: {args.ppe_checkpoint}")
    ppe_model, keypoint_model = load_models(args.ppe_checkpoint)
    if args.image is not None:
        infer_image(args, ppe_model, keypoint_model)
    else:
        infer_video(args, ppe_model, keypoint_model)


if __name__ == "__main__":
    main()
