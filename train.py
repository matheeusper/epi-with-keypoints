import os
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import torch
import numpy as np
import random
import yaml
import argparse
from rfdetr import RFDETRSmall
from rfdetr.detr import RFDETR


def set_seed(seed: int = 42):
    """Fixa as sementes para garantir a reprodutibilidade dos experimentos."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # para múltiplas GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_class_names(dataset_dir: Path) -> dict[int, str]:
    data_file = dataset_dir / "data.yaml"
    if not data_file.is_file():
        raise FileNotFoundError(f"Arquivo de classes não encontrado: {data_file}")
    data = yaml.safe_load(data_file.read_text(encoding="utf-8")) or {}
    raw_names = data.get("names")
    if isinstance(raw_names, list):
        names = {index: str(name) for index, name in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        names = {int(class_id): str(name) for class_id, name in raw_names.items()}
    else:
        raise ValueError(f"Campo 'names' inválido em {data_file}: esperado lista ou mapa.")
    if not names:
        raise ValueError(f"Nenhuma classe declarada em {data_file}.")
    return names


def resolve_classes(requested: list[str], names: dict[int, str]) -> list[int]:
    tokens = [part.strip() for item in requested for part in item.split(",") if part.strip()]
    if any(token.casefold() == "all" for token in tokens):
        if len(tokens) != 1:
            raise ValueError("Use 'all' sozinho, sem combinar com outras classes.")
        return sorted(names)

    name_to_id = {name.casefold(): class_id for class_id, name in names.items()}
    selected: list[int] = []
    for token in tokens:
        if token.isdigit():
            class_id = int(token)
            if class_id not in names:
                raise ValueError(f"ID de classe inexistente: {class_id}.")
        else:
            try:
                class_id = name_to_id[token.casefold()]
            except KeyError as error:
                available = ", ".join(f"{class_id}:{name}" for class_id, name in sorted(names.items()))
                raise ValueError(f"Classe desconhecida: {token!r}. Disponíveis: {available}") from error
        if class_id not in selected:
            selected.append(class_id)
    if not selected:
        raise ValueError("Informe ao menos uma classe ou use --classes all.")
    return selected


@contextmanager
def select_yolo_classes(
    selected_ids: list[int],
    names: dict[int, str],
    negative_ratio: float | None = None,
    seed: int = 42,
):
    """Filtra e reindexa as classes em memória, sem alterar ou copiar o dataset."""
    from rfdetr.datasets import yolo as yolo_module

    original_builder = yolo_module._build_yolo_samples
    original_detector = RFDETR._detect_num_classes_for_training
    mapping = {source_id: new_id for new_id, source_id in enumerate(selected_ids)}
    selected_names = [names[source_id] for source_id in selected_ids]

    def filtered_builder(*args, **kwargs):
        _, samples = original_builder(*args, **kwargs)
        filtered_samples = []
        for sample in samples:
            indexes = np.flatnonzero(np.isin(sample.class_id, selected_ids))
            remapped_ids = np.array(
                [mapping[int(sample.class_id[index])] for index in indexes], dtype=np.int64
            )
            polygons = (
                tuple(sample.polygons[index] for index in indexes)
                if len(sample.polygons) == len(sample.class_id)
                else sample.polygons
            )
            keypoints = (
                sample.keypoints[indexes]
                if len(sample.keypoints) == len(sample.class_id)
                else sample.keypoints
            )
            filtered_samples.append(
                replace(
                    sample,
                    xyxy=sample.xyxy[indexes],
                    class_id=remapped_ids,
                    polygons=polygons,
                    keypoints=keypoints,
                )
            )

        image_folder = Path(args[0] if args else kwargs["img_folder"])
        split = image_folder.parent.name
        positives = [sample for sample in filtered_samples if len(sample.class_id) > 0]
        negatives = [sample for sample in filtered_samples if len(sample.class_id) == 0]
        if split == "train" and negative_ratio is not None:
            if not positives:
                raise ValueError("Nenhuma imagem positiva encontrada para as classes selecionadas.")
            negative_limit = min(len(negatives), round(len(positives) * negative_ratio))
            rng = random.Random(seed)
            sampled_negatives = rng.sample(negatives, negative_limit)
            filtered_samples = sorted(
                positives + sampled_negatives, key=lambda sample: sample.image_path
            )
            print(
                f"Amostragem do treino: {len(positives)} imagens positivas + "
                f"{len(sampled_negatives)} negativas (proporção {negative_ratio:g}:1)."
            )
        elif split != "train":
            print(
                f"Split {split}: {len(positives)} imagens positivas + "
                f"{len(negatives)} negativas (avaliação completa)."
            )
        return selected_names, filtered_samples

    def selected_class_count(dataset_dir: str, *, use_grouppose_keypoints: bool = False) -> int:
        return len(selected_ids)

    yolo_module._build_yolo_samples = filtered_builder
    RFDETR._detect_num_classes_for_training = staticmethod(selected_class_count)
    try:
        yield selected_names
    finally:
        yolo_module._build_yolo_samples = original_builder
        RFDETR._detect_num_classes_for_training = staticmethod(original_detector)


def main():
    parser = argparse.ArgumentParser(description="Treinar o modelo RFDETRSmall com configurações YAML.")
    parser.add_argument("--config", type=str, default="configs/base.yaml",
                        help="Caminho para o arquivo de configuração YAML.")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["all"],
        help="Classes para treinar, por nome ou ID. Exemplos: --classes helmet; --classes helmet Vest; --classes all.",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=1.0,
        help=(
            "Quantidade máxima de imagens negativas por imagem positiva quando classes específicas "
            "são escolhidas. Padrão: 1.0. Use -1 para manter todas."
        ),
    )
    args = parser.parse_args()
    if args.negative_ratio < 0 and args.negative_ratio != -1:
        parser.error("--negative-ratio deve ser maior ou igual a zero, ou -1 para manter todas.")

    CONFIG_FILE = args.config

    # Carrega as configurações do arquivo YAML
    with open(CONFIG_FILE, 'r') as f:
        config_data = yaml.safe_load(f)

    # Extrai as configurações para passar ao modelo
    general_config = config_data.get("general", {})
    model_config = config_data.get("model", {})
    training_config = config_data.get("training", {})
    early_stopping_config = config_data.get("early_stopping", {})
    aug_config = config_data.get("augmentations", {})

    # Mapeia as configurações do YAML para os argumentos esperados por model.train
    train_args = {
        "output_dir": general_config.get("output_dir"),
        "dataset_dir": general_config.get("dataset_dir"),
        "device": general_config.get("device"),
        "epochs": training_config.get("epochs"),
        "batch_size": training_config.get("batch_size"),
        "grad_accum_steps": training_config.get("grad_accum_steps"),
        "lr": training_config.get("lr"),
        "lr_encoder": training_config.get("lr_encoder"),
        "early_stopping": early_stopping_config.get("enabled"),
        "early_stopping_patience": early_stopping_config.get("patience"),
        "early_stopping_min_delta": early_stopping_config.get("min_delta"),
        "early_stopping_use_ema": early_stopping_config.get("use_ema"),
        "aug_config": aug_config,
    }

    # 1. Preparar ambiente
    os.makedirs(train_args["output_dir"], exist_ok=True)
    set_seed(general_config.get("seed"))

    # 2. Inicializar o modelo.
    # `resolution` e `gradient_checkpointing` pertencem à configuração do modelo,
    # portanto devem ser passados ao construtor — não ao método `train()`.
    model_args = {
        "resolution": model_config.get("resolution"),
        "gradient_checkpointing": model_config.get("gradient_checkpointing"),
    }
    model = RFDETRSmall(**{key: value for key, value in model_args.items() if value is not None})

    # 3. Selecionar/reindexar as classes apenas em memória.
    source_dataset = Path(train_args["dataset_dir"])
    names = load_class_names(source_dataset)
    selected_ids = resolve_classes(args.classes, names)
    selection = ", ".join(
        f"{source_id}:{names[source_id]} -> {new_id}"
        for new_id, source_id in enumerate(selected_ids)
    )
    print(f"Classes do treinamento: {selection}")

    selecting_all = selected_ids == sorted(names)
    negative_ratio = None if selecting_all or args.negative_ratio == -1 else args.negative_ratio
    with select_yolo_classes(
        selected_ids,
        names,
        negative_ratio=negative_ratio,
        seed=general_config.get("seed", 42),
    ) as selected_names:
        train_args["class_names"] = selected_names
        model.train(**train_args)

if __name__ == "__main__":
    main()
