"""Deriva um dataset YOLO de uma classe a partir do Construction-PPE.

O resultado segue o layout esperado pelo RF-DETR para datasets YOLO:
``<saida>/<split>/{images,labels}``, com a classe ``helmet`` reindexada como 0.
Por padrão as imagens são links simbólicos para não duplicar o dataset de origem.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

import yaml


SOURCE_ROOT = Path("construction-ppe")
TARGET_ROOT = Path("construction-ppe-helmet")
SOURCE_CLASS_HELMET = 0
SPLITS = {"train": ("train",), "valid": ("valid", "val"), "test": ("test",)}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cria um dataset YOLO contendo apenas a classe helmet."
    )
    parser.add_argument("--source", type=Path, default=SOURCE_ROOT, help="Dataset Construction-PPE original.")
    parser.add_argument("--output", type=Path, default=TARGET_ROOT, help="Diretório do dataset derivado.")
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copia imagens em vez de criar links simbólicos.",
    )
    return parser.parse_args()


def resolve_split_directory(source: Path, split_names: tuple[str, ...], kind: str) -> Path:
    """Aceita tanto o layout do projeto quanto o layout oficial do Ultralytics."""
    for split in split_names:
        for candidate in (source / split / kind, source / kind / split):
            if candidate.is_dir():
                return candidate
    names = ", ".join(split_names)
    raise FileNotFoundError(f"Não foi encontrado {kind} para o split {names} em {source}")


def image_for_label(label: Path, images_dir: Path) -> Path:
    # Alguns exports possuem rótulos duplicados como ``image940(1).txt``,
    # mantendo a imagem original como ``image940.jpg``.
    stems = (label.stem, re.sub(r"\(\d+\)$", "", label.stem))
    for stem in stems:
        for suffix in IMAGE_SUFFIXES:
            candidate = images_dir / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"Imagem correspondente a {label.name} não encontrada em {images_dir}")


def helmet_lines(label: Path) -> list[str]:
    """Mantém apenas caixas da classe 0 (helmet) e valida o formato YOLO."""
    selected: list[str] = []
    for line in label.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Rótulo YOLO inválido em {label}: {line!r}")
        if int(fields[0]) == SOURCE_CLASS_HELMET:
            selected.append("0 " + " ".join(fields[1:]))
    return selected


def is_duplicate_export(label: Path) -> bool:
    """Ignora cópias de rótulo ``nome(1).txt`` quando o original existe."""
    base_stem = re.sub(r"\(\d+\)$", "", label.stem)
    return base_stem != label.stem and (label.parent / f"{base_stem}.txt").is_file()


def link_or_copy(source: Path, target: Path, copy_images: bool) -> None:
    """Cria a imagem derivada sem substituir arquivo existente de outro destino."""
    if target.exists() or target.is_symlink():
        if target.resolve() == source.resolve():
            return
        if copy_images and target.is_file() and target.stat().st_size == source.stat().st_size:
            return
        raise FileExistsError(f"Destino já existe e não corresponde à origem: {target}")
    if copy_images:
        shutil.copy2(source, target)
    else:
        target.symlink_to(os.path.relpath(source, target.parent))


def main() -> None:
    args = parse_args()
    if not args.source.is_dir():
        raise FileNotFoundError(f"Dataset de origem não encontrado: {args.source}")
    totals = {"images": 0, "helmet_annotations": 0}
    for output_split, input_splits in SPLITS.items():
        images_dir = resolve_split_directory(args.source, input_splits, "images")
        labels_dir = resolve_split_directory(args.source, input_splits, "labels")
        output_images = args.output / output_split / "images"
        output_labels = args.output / output_split / "labels"
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)
        labels = [label for label in sorted(labels_dir.glob("*.txt")) if not is_duplicate_export(label)]
        if not labels:
            raise FileNotFoundError(f"Nenhum rótulo encontrado em: {labels_dir}")
        for label in labels:
            image = image_for_label(label, images_dir)
            # O nome de saída segue o rótulo para o pareamento YOLO, inclusive
            # nos casos de rótulos com sufixo ``(1)``.
            link_or_copy(image, output_images / f"{label.stem}{image.suffix}", args.copy_images)
            selected = helmet_lines(label)
            (output_labels / label.name).write_text(
                "\n".join(selected) + ("\n" if selected else ""), encoding="utf-8"
            )
            totals["images"] += 1
            totals["helmet_annotations"] += len(selected)

    data_yaml = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": {0: "helmet"},
    }
    (args.output / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(f"Dataset derivado salvo em: {args.output}")
    print(f"Imagens: {totals['images']}; anotações de capacete: {totals['helmet_annotations']}")


if __name__ == "__main__":
    main()
