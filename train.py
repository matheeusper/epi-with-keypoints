import os
from rfdetr import RFDETRMedium

output_path = "/home/matheusandrade/Documentos/epi-with-keypoints/output"
os.makedirs(output_path, exist_ok=True)

# 1. Inicializar o modelo
model = RFDETRMedium()

# 2. Rodar o treinamento
model.train(
    dataset_dir="/home/matheusandrade/Documentos/epi-with-keypoints/construction-ppe",  # Caminho do dataset
    epochs=100,
    output_dir=output_path,
    
    batch_size=4,                  
    grad_accum_steps=4,            
    gradient_checkpointing=True,   
    lr=1e-4,                      
    
    # --- Early Stopping ---
    early_stopping=True,
    early_stopping_patience=12,    # Espera 12 épocas sem melhoria antes de parar
    early_stopping_min_delta=0.001, # Exige ganho mínimo de 0.1% no box mAP
    early_stopping_use_ema=True,   # Mede a performance no modelo EMA
    
    # --- Augmentations Específicas para Detecção de EPIs ---
    aug_config={
        "HorizontalFlip": {"p": 0.5},
        "RandomBrightnessContrast": {"p": 0.4},  # Simula variações de iluminação interna/externa
        "ShiftScaleRotate": {                   # Pequenas variações de escala e ângulo de câmera
            "shift_limit": 0.05, 
            "scale_limit": 0.1, 
            "rotate_limit": 10, 
            "p": 0.5
        }
    }
)