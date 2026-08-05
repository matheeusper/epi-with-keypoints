"""Valida e publica a release do modelo no Hugging Face Hub com o CLI ``hf``."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_REPO = "matheeusper/epi-with-keypoints"
DEFAULT_MODEL_PATH = ROOT / "models" / "helmet_only_576_best.pth"
MODEL_CARD = ROOT / "huggingface" / "README.md"
MODEL_METADATA = ROOT / "huggingface" / "model_metadata.json"
TRAINING_CONFIG = ROOT / "configs" / "config_helmet_only_576.yaml"


def sha256(path: Path) -> str:
    """Calcula o checksum do arquivo em blocos, sem carregá-lo inteiro na memória."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_release(
    checkpoint: Path,
    metadata_path: Path = MODEL_METADATA,
    model_card: Path = MODEL_CARD,
    training_config: Path = TRAINING_CONFIG,
) -> dict:
    """Confere manifesto, tamanho e checksum antes de qualquer upload."""
    required = (checkpoint, metadata_path, model_card, training_config)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Arquivos obrigatórios ausentes: " + ", ".join(missing))

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    checkpoint_metadata = metadata["checkpoint"]
    if checkpoint.name != checkpoint_metadata["filename"]:
        raise ValueError("O nome do checkpoint diverge de model_metadata.json.")
    if checkpoint.stat().st_size != checkpoint_metadata["size_bytes"]:
        raise ValueError("O tamanho do checkpoint diverge de model_metadata.json.")
    actual_sha = sha256(checkpoint)
    if actual_sha != checkpoint_metadata["sha256"]:
        raise ValueError("O SHA-256 do checkpoint diverge de model_metadata.json.")
    return metadata


def stage_release(checkpoint: Path, destination: Path) -> None:
    """Monta exatamente os quatro arquivos permitidos no repositório de modelo."""
    shutil.copy2(MODEL_CARD, destination / "README.md")
    shutil.copy2(MODEL_METADATA, destination / "model_metadata.json")
    shutil.copy2(TRAINING_CONFIG, destination / "training_config.yaml")
    try:
        os.link(checkpoint, destination / checkpoint.name)
    except OSError:
        shutil.copy2(checkpoint, destination / checkpoint.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valida e publica checkpoint, model card, config e metadados no Hub."
    )
    parser.add_argument("--repo-id", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--commit-message",
        default="Atualiza release do modelo RF-DETR de capacetes",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Valida todos os artefatos sem acessar ou modificar o Hub.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    metadata = validate_release(checkpoint)
    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "checkpoint": str(checkpoint),
                "size_bytes": checkpoint.stat().st_size,
                "sha256": metadata["checkpoint"]["sha256"],
                "dry_run": args.dry_run,
            },
            indent=2,
        )
    )
    if args.dry_run:
        return
    if shutil.which("hf") is None:
        raise RuntimeError("CLI `hf` não encontrado. Instale e execute `hf auth login`.")

    with tempfile.TemporaryDirectory(prefix=".hf-release-", dir=ROOT) as temp_dir:
        release_dir = Path(temp_dir)
        stage_release(checkpoint, release_dir)
        subprocess.run(
            [
                "hf",
                "upload",
                args.repo_id,
                str(release_dir),
                ".",
                "--type",
                "model",
                "--commit-message",
                args.commit_message,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
