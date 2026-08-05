---
license: agpl-3.0
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
  - Ultralytics/Construction-PPE
---

# RF-DETR Small — detecção de capacetes em obras

Checkpoint do `RFDETRSmall` treinado para detectar exclusivamente a classe positiva
`helmet` em imagens de canteiros de obras. O modelo integra o projeto
[epi-with-keypoints](https://github.com/matheeusper/epi-with-keypoints), que combina
a detecção de capacete com keypoints corporais e tracking temporal para produzir
um resultado por pessoa em imagens e vídeos.

Este repositório no Hugging Face contém apenas a release do modelo. O código de
treinamento, avaliação, inferência, tracking, testes e a documentação completa são
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

Avaliação em 141 imagens e 192 capacetes do split de teste derivado do
Construction-PPE:

| Métrica | Resultado |
| --- | ---: |
| mAP@[0.50:0.95] | 55,53% |
| mAP@0.50 | 95,00% |
| mAP@0.75 | 58,66% |
| mAR@100 | 66,30% |
| Precisão@0.50 | 93,75% |
| Recall@0.50 | 93,75% |
| F1@0.50 | 93,75% |
| Confiança no melhor F1 | 0,435 |

## Treinamento

- Arquitetura: `RFDETRSmall`.
- Resolução: 576 px.
- Classe: `helmet` (`id=0`).
- Seed: 42.
- Batch efetivo: 16 (`batch_size=1`, `grad_accum_steps=16`).
- Checkpoint selecionado após 11 épocas e 781 passos globais.

## Limitações e uso responsável

O modelo foi treinado em um conjunto específico de obras e pode falhar com pessoas
pequenas, iluminação extrema, ângulos incomuns, aglomerações ou oclusões prolongadas.
Ele detecta capacetes, mas não deve ser usado isoladamente para decisões disciplinares
ou outras decisões críticas de segurança. Valide e calibre o limiar com imagens do
ambiente de implantação.

O dataset de origem Construction-PPE é distribuído sob AGPL-3.0; este repositório
adota a mesma licença para o checkpoint derivado.
