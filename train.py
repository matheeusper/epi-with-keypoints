import os
import torch
import numpy as np
import random
import yaml
import argparse
from rfdetr import RFDETRSmall


def set_seed(seed: int = 42):
    """Fixa as sementes para garantir a reprodutibilidade dos experimentos."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # para múltiplas GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Treinar o modelo RFDETRSmall com configurações YAML.")
    parser.add_argument("--config", type=str, default="config_base.yaml",
                        help="Caminho para o arquivo de configuração YAML.")
    args = parser.parse_args()

    CONFIG_FILE = args.config

    # Carrega as configurações do arquivo YAML
    with open(CONFIG_FILE, 'r') as f:
        config_data = yaml.safe_load(f)

    # Extrai as configurações para passar ao modelo
    general_config = config_data.get("general", {})
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
        "gradient_checkpointing": training_config.get("gradient_checkpointing"),
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

    # 2. Inicializar e treinar o modelo
    model = RFDETRSmall() # Assumindo que RFDETRSmall não precisa de argumentos no construtor
    model.train(**train_args)

if __name__ == "__main__":
    main()