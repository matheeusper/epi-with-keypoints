"""Localiza o checkpoint local ou baixa o modelo publicado no Hugging Face Hub."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download


DEFAULT_MODEL_REPO = "matheeusper/epi-with-keypoints"
DEFAULT_MODEL_FILENAME = "helmet_only_576_best.pth"
DEFAULT_MODEL_PATH = Path("models") / DEFAULT_MODEL_FILENAME


def resolve_checkpoint(
    checkpoint: Path,
    repo_id: str = DEFAULT_MODEL_REPO,
    default_path: Path = DEFAULT_MODEL_PATH,
) -> Path:
    """Retorna um checkpoint local, baixando o padrão do Hub quando necessário."""
    if checkpoint.is_file():
        return checkpoint
    if checkpoint != default_path:
        raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint}")

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint local ausente; baixando {repo_id}/{DEFAULT_MODEL_FILENAME}...")
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=DEFAULT_MODEL_FILENAME,
        local_dir=checkpoint.parent,
    )
    return Path(downloaded)
