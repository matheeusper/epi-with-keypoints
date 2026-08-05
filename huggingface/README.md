---
license: cc-by-4.0
library_name: rfdetr
pipeline_tag: object-detection
tags:
  - computer-vision
  - object-detection
  - construction-safety
  - ppe
  - helmet-detection
  - rf-detr
datasets:
  - 51ddhesh/PPE_Detection
model-index:
  - name: RF-DETR Small Helmet 576
    results:
      - task:
          type: object-detection
          name: Object Detection
        dataset:
          name: PPE_Detection
          type: 51ddhesh/PPE_Detection
          split: test
        metrics:
          - type: map
            value: 0.5890050577
            name: mAP@50:95
          - type: map
            value: 0.8961793948
            name: mAP@50
---

# RF-DETR Small — detecção de capacetes em obras

Checkpoint do `RFDETRSmall` treinado para detectar exclusivamente a classe positiva
`helmet` em imagens de canteiros de obras. O modelo integra o projeto
[epi-with-keypoints](https://github.com/matheeusper/epi-with-keypoints), que combina
a detecção de capacete com keypoints corporais e tracking temporal para produzir
um resultado por pessoa em imagens e vídeos.

Este repositório no Hugging Face contém apenas a release do modelo. O código de
treinamento, avaliação, inferência, tracking e a documentação completa são
mantidos no [repositório GitHub](https://github.com/matheeusper/epi-with-keypoints).

## Arquivo

- `helmet_only_576_best.pth`: checkpoint RF-DETR em resolução 576 px.
- `training_config.yaml`: configuração usada no treinamento.
- `model_metadata.json`: classes, limiares recomendados, métricas e checksum.

## Uso

Clone o projeto e sincronize as dependências:

```bash
git clone https://github.com/matheeusper/epi-with-keypoints.git
cd epi-with-keypoints
uv sync
```

O script baixa este checkpoint automaticamente quando `models/helmet_only_576_best.pth`
não está disponível:

```bash
uv run python infer_ppe_keypoints.py --image images/image2.jpg
uv run python infer_ppe_keypoints.py --video videos/obra.mp4
```

Também é possível baixá-lo diretamente:

```python
from huggingface_hub import hf_hub_download

checkpoint = hf_hub_download(
    repo_id="matheeusper/epi-with-keypoints",
    filename="helmet_only_576_best.pth",
)
```

```python
from rfdetr import RFDETRSmall

model = RFDETRSmall.from_checkpoint(checkpoint)
detections = model.predict("imagem.jpg", threshold=0.35)
```

## Métricas

Avaliação em 1.234 imagens e 597 capacetes do split de teste completo do
PPE_Detection, incluindo imagens negativas sem capacete:

| Métrica | Resultado |
| --- | ---: |
| mAP@[0.50:0.95] | 58,90% |
| mAP@0.50 | 89,62% |
| mAP@0.75 | 67,41% |
| mAR@100 | 74,09% |
| Precisão@0.50 | 86,68% |
| Recall@0.50 | 90,45% |
| F1@0.50 | 88,52% |
| Confiança no melhor F1 | 0,570 |

## Treinamento

- Arquitetura: `RFDETRSmall`.
- Resolução: 576 px.
- Classe: `helmet` (`id=0`).
- Seed: 42.
- Batch efetivo: 16 (`batch_size=4`, `grad_accum_steps=4`).
- Checkpoint regular selecionado na época 19, após 20 épocas e 2.000 passos globais.

## Limitações e uso responsável

O modelo foi treinado em um conjunto específico de obras e pode falhar com pessoas
pequenas, iluminação extrema, ângulos incomuns, aglomerações ou oclusões prolongadas.
Ele detecta capacetes, mas não deve ser usado isoladamente para decisões disciplinares
ou outras decisões críticas de segurança. Valide e calibre o limiar com imagens do
ambiente de implantação.

O dataset de origem
[51ddhesh/PPE_Detection](https://huggingface.co/datasets/51ddhesh/PPE_Detection)
é distribuído sob CC BY 4.0; o checkpoint derivado é publicado sob a mesma licença.
