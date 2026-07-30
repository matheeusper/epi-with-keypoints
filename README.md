# Detecção e validação de capacetes com RF-DETR

Projeto de visão computacional para canteiros de obras. Ele treina um `RFDETRSmall` para detectar EPIs e o combina com `RFDETRKeypointPreview` para validar o uso de capacetes por pessoa.

## Funcionamento

```text
Imagem
├── RFDETRSmall treinado ───────────────► capacetes
└── RFDETRKeypointPreview pré-treinado ─► pessoas e keypoints
                                             │
                                             ▼
                       associação + validação da região da cabeça
                                             │
                                             ▼
                       imagem anotada + relatório JSON
```

O detector treinado pode conter outras classes, mas a pipeline atual usa somente a classe positiva `helmet` ou `capacete`. As classes `person`, `none`, fundo e demais EPIs são ignoradas. As pessoas são obtidas exclusivamente pelo modelo de keypoints.

## Recursos

- Treinamento configurável por YAML.
- Receitas para treino padrão, rápido, com aumentos agressivos e learning rate menor.
- Inferência combinada de capacete e pose humana.
- Associação automática do capacete à pessoa mais próxima.
- Validação geométrica pela região da cabeça.
- Alertas minimalistas e relatório JSON por pessoa.

## Instalação

Requisitos: Python 3.10+, [uv](https://docs.astral.sh/uv/) e, preferencialmente, GPU CUDA.

```bash
git clone git@github.com:matheeusper/epi-with-keypoints.git
cd epi-with-keypoints
uv sync
```

Os comandos usam `uv run`, portanto não é preciso ativar o ambiente virtual manualmente.

## Dataset

O caminho padrão é `construction-ppe/`, configurável nos arquivos YAML. RF-DETR aceita COCO e YOLO. Para COCO, use:

```text
construction-ppe/
├── train/
│   ├── _annotations.coco.json
│   └── imagens...
├── valid/
│   ├── _annotations.coco.json
│   └── imagens...
└── test/                       # opcional
    ├── _annotations.coco.json
    └── imagens...
```

Para a validação atual, rotule a classe positiva de capacete como `helmet` ou `capacete`.

## Treinamento

O [train.py](train.py) carrega a receita, fixa a seed, instancia `RFDETRSmall` e inicia o treinamento.

```bash
uv run python train.py --config configs/config_base.yaml
```

| Receita | Uso |
| --- | --- |
| `configs/config_base.yaml` | Treinamento padrão, até 100 épocas. |
| `configs/config_fast_train.yaml` | Teste rápido do pipeline. |
| `configs/config_high_aug.yaml` | Aumentos de dados mais fortes. |
| `configs/config_low_lr.yaml` | Ajuste fino com learning rate menor. |
| `configs/config_helmet_smoke_576.yaml` | Teste de VRAM e pipeline em 576 px. |
| `configs/config_helmet_only_576.yaml` | Treinamento de alta qualidade, apenas capacete. |

As receitas contêm `general`, `training`, `early_stopping` e `augmentations`. Ajuste `dataset_dir`, `device`, `epochs`, `batch_size` e `lr` conforme seu ambiente.

O melhor checkpoint é salvo no diretório configurado. Por padrão, a inferência usa `outputSmall/checkpoint_best_total.pth`.

## Inferência e validação de capacete

O [infer_ppe_keypoints.py](infer_ppe_keypoints.py) carrega o checkpoint treinado, o modelo de keypoints e a imagem. Na primeira execução, os pesos de keypoints podem ser baixados automaticamente.

```bash
uv run python infer_ppe_keypoints.py --image images/image.jpg
```

Por padrão, serão criados ao lado da entrada:

```text
images/
├── image_ppe_keypoints.jpg   # imagem anotada
└── image_ppe_keypoints.json  # relatório de conformidade
```

### Anotações

- `P1 ✓` verde: pessoa com capacete validado na região da cabeça.
- `P1 !` vermelho: pessoa sem capacete validado.
- `helmet OK` verde: capacete em posição compatível.
- `helmet X` vermelho: capacete fora da região esperada ou sem associação.

Os keypoints são usados internamente e ficam ocultos por padrão para manter a imagem limpa.

### Opções

```bash
uv run python infer_ppe_keypoints.py \
  --image images/image.jpg \
  --ppe-checkpoint outputSmall/checkpoint_best_total.pth \
  --output resultados/imagem_anotada.jpg \
  --report resultados/relatorio.json \
  --ppe-threshold 0.35 \
  --keypoint-threshold 0.55
```

| Opção | Descrição |
| --- | --- |
| `--image` | Imagem de entrada; obrigatória. |
| `--ppe-checkpoint` | Checkpoint do detector de EPIs. |
| `--output` | Caminho da imagem anotada. |
| `--report` | Caminho do relatório JSON. |
| `--ppe-threshold` | Limiar de capacete; padrão `0.35`. |
| `--keypoint-threshold` | Limiar de pessoas; padrão `0.55`. |
| `--keypoint-confidence` | Confiança mínima de keypoint; padrão `0.30`. |
| `--draw-keypoints` | Exibe keypoints para depuração. |
| `--hide-person-boxes` | Oculta as caixas de pessoas. |

## Relatório JSON

O relatório registra a validação por pessoa:

```json
{
  "pessoas": [
    {
      "id": 1,
      "epis_validados": 1,
      "epis": { "capacete": { "status": "validado" } }
    }
  ]
}
```

Estados possíveis: `validado`, `fora_da_regiao`, `nao_verificavel` e `ausente`.

## Estrutura

```text
.
├── train.py                     # treinamento do RFDETRSmall
├── infer_ppe_keypoints.py       # inferência e validação de capacete
├── configs/                     # receitas de treinamento YAML
├── outputSmall/                 # checkpoint local, não versionado
├── construction-ppe/            # dataset local, não versionado
├── images/                      # imagens de exemplo
└── docs-rfdetr/                 # referência local do RF-DETR
```

## Limitações

- A regra atual avalia exclusivamente capacete.
- Pessoas pequenas, oclusas ou com keypoints imprecisos podem gerar resultado inconclusivo.
- Calibre os limiares com imagens reais do ambiente de operação.
- Resultados devem ser revisados antes de decisões críticas de segurança.
