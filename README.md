# Detecção e validação de capacetes com RF-DETR

Projeto de visão computacional para canteiros de obras. Ele treina um `RFDETRSmall`
para detectar capacetes e o combina com `RFDETRKeypointPreview` para validar o uso
do capacete por pessoa.

**Código, testes e receitas:** este repositório no
[GitHub](https://github.com/matheeusper/epi-with-keypoints) · **Checkpoint treinado
e model card:** [Hugging Face Hub](https://huggingface.co/matheeusper/epi-with-keypoints)

O GitHub não armazena pesos, datasets ou vídeos completos. O Hugging Face não
duplica o código da aplicação: ele mantém somente a release reproduzível do
modelo. A fronteira e o processo de publicação estão documentados em
[docs/PUBLISHING.md](docs/PUBLISHING.md).

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

Em vídeos, as caixas de pessoa passam também pelo ByteTrack antes da associação.
O tracker mantém um ID estável para cada pessoa e reaproveita o estado durante
oclusões curtas; imagens continuam sendo avaliadas de forma independente.

O checkpoint deste projeto possui uma única classe positiva, `helmet`. As pessoas
são obtidas exclusivamente pelo modelo pré-treinado de keypoints.

### Por que combinar um modelo treinado com keypoints pré-treinados?

O `RFDETRSmall` é treinado especificamente com as imagens e anotações de EPI do
projeto. Por isso, ele aprende a localizar capacetes no cenário real de canteiro,
com iluminação, distância, ângulos e tipos de equipamento presentes no dataset.
Por outro lado, ele não oferece contexto corporal suficiente para responder a
pergunta mais importante: **aquele capacete pertence a qual pessoa?**

O `RFDETRKeypointPreview` já vem pré-treinado para localizar pessoas e os pontos
da cabeça, como nariz, olhos e orelhas. Usá-lo evita a necessidade de anotar
keypoints manualmente no dataset de EPI e permite validar se o capacete detectado
está na região esperada da pessoa.

Essa combinação traz três vantagens principais:

- **Menos falsos positivos:** um objeto parecido com capacete, mas longe de uma
  cabeça, pode ser marcado como inválido.
- **Resultado por pessoa:** em vez de apenas contar capacetes na imagem, a
  pipeline informa quais pessoas estão protegidas ou em alerta.
- **Melhor aproveitamento do dataset:** o detector é especializado no EPI local,
  enquanto o conhecimento corporal vem de um modelo de pose já treinado em grande
  escala.

## Recursos

- Treinamento configurável por YAML.
- Receitas para treino padrão, rápido, com aumentos agressivos e learning rate menor.
- Inferência combinada de capacete e pose humana.
- Associação automática do capacete à pessoa mais próxima.
- Tracking de pessoas com IDs persistentes em vídeos.
- Estabilização temporal dos alertas de ausência de capacete.
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

O dataset de origem é o [Construction-PPE da Ultralytics](https://docs.ultralytics.com/datasets/detect/construction-ppe/), disponibilizado sob licença AGPL-3.0. Ele contém 1.416 imagens divididas em 1.132 de treino, 143 de validação e 141 de teste, originalmente anotadas em 11 classes de EPI.

Este projeto não treina as 11 classes. O script
[`prepare_helmet_dataset.py`](prepare_helmet_dataset.py) mantém somente as caixas
da classe original `helmet` (id `0`), descarta as demais — inclusive `Person` e
`no_helmet` — e produz um dataset de uma única classe, também id `0`.

O resultado é normalizado para o layout YOLO aceito pelo RF-DETR:

```text
construction-ppe-helmet/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Baixe e descompacte o dataset oficial como `construction-ppe/` na raiz do projeto.
Em seguida, gere a versão de capacete:

```bash
uv run python prepare_helmet_dataset.py
```

Por padrão, as imagens do dataset derivado são links simbólicos para evitar
duplicação. Para uma cópia independente, use `--copy-images`. O preparador aceita
tanto o layout oficial `images/<split>` quanto o layout local `<split>/images`.

## Treinamento

O [train.py](train.py) carrega a receita, fixa a seed, instancia `RFDETRSmall` e inicia o treinamento.

```bash
uv run python train.py --config configs/config_helmet_only_576.yaml
```

| Receita | Uso |
| --- | --- |
| `configs/config_base.yaml` | Receita legada para o dataset multiclasse. |
| `configs/config_fast_train.yaml` | Teste rápido para o dataset multiclasse. |
| `configs/config_high_aug.yaml` | Aumentos fortes para o dataset multiclasse. |
| `configs/config_low_lr.yaml` | Ajuste fino multiclasse com learning rate menor. |
| `configs/config_helmet_smoke_576.yaml` | Teste de VRAM e pipeline em 576 px. |
| `configs/config_helmet_only_576.yaml` | Receita recomendada: treino final somente de capacete. |

As receitas contêm `general`, `training`, `early_stopping` e `augmentations`. Ajuste `dataset_dir`, `device`, `epochs`, `batch_size` e `lr` conforme seu ambiente.
Elas usam `device: cuda`; sem GPU CUDA, altere esse campo para `cpu` (o treino será
consideravelmente mais lento).

O treino recomendado grava o melhor modelo em
`output_helmet_only_576/checkpoint_best_total.pth`. Pesos não são versionados no
GitHub. O checkpoint publicado está disponível em
[matheeusper/epi-with-keypoints](https://huggingface.co/matheeusper/epi-with-keypoints)
no Hugging Face Hub. Se `models/helmet_only_576_best.pth` não existir, os scripts
de inferência e avaliação baixam esse arquivo automaticamente. Também é possível
informar um checkpoint local com `--ppe-checkpoint` ou outro repositório com
`--hf-repo-id`.

## Métricas do treinamento

O checkpoint avaliado (`models/helmet_only_576_best.pth`) foi treinado para a única
classe `helmet`, em resolução de 576 px. Os rótulos do conjunto derivado
registram 1.132 exemplos de treino, 143 de validação e 141 de teste; respectivamente,
1.341, 201 e 192 anotações de capacete.

| Indicador do checkpoint | Valor |
| --- | --- |
| Épocas concluídas no checkpoint | 11 |
| Passos globais | 781 |

### Avaliação no conjunto de teste

O checkpoint foi avaliado em 141 imagens e 192 capacetes anotados do split `test`.
As métricas abaixo usam a avaliação COCO de caixas; precisão, recall e F1 usam
IoU de 0,50 no melhor limiar de confiança.

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

Para reproduzir a avaliação, execute:

```bash
uv run python evaluate_helmet.py \
  --checkpoint output_helmet_only_576/checkpoint_best_total.pth
```

O relatório completo é salvo em `outputs/evaluation/helmet_test_metrics.json`.
O avaliador usa `construction-ppe-helmet/test`, já preparado com imagens e rótulos
da única classe `helmet`.

## Inferência e validação de capacete

O [infer_ppe_keypoints.py](infer_ppe_keypoints.py) carrega o checkpoint treinado, o modelo de keypoints e a imagem. Na primeira execução, os pesos de keypoints podem ser baixados automaticamente.

```bash
uv run python infer_ppe_keypoints.py \
  --image images/image2.jpg \
  --ppe-checkpoint output_helmet_only_576/checkpoint_best_total.pth
```

Sem `--ppe-checkpoint`, o modelo é usado de `models/` ou baixado automaticamente
do Hugging Face Hub na primeira execução:

```bash
uv run python infer_ppe_keypoints.py --image images/image2.jpg
```

Também é possível processar um vídeo completo. Os modelos são carregados uma única
vez e aplicados a cada quadro; a saída preserva a taxa de quadros do arquivo de
entrada. Por padrão, identidades oclusas são conservadas por até 1 segundo e uma
ausência só vira alerta quando ocupa a maioria estrita de uma janela de 1 segundo.
Detecções de pessoa abaixo do limiar principal podem prolongar um track existente,
mas não são desenhadas nem contam como ausência de capacete.

```bash
uv run python infer_ppe_keypoints.py --video videos/obra.mp4
```

O comando acima cria `outputs/obra/annotated/obra_ppe_keypoints.mp4` e
`outputs/obra/reports/obra_ppe_keypoints.json`. O JSON inclui o resultado de cada
quadro e o respectivo instante em segundos. Para escolher os destinos e testar
apenas parte do vídeo:

```bash
uv run python infer_ppe_keypoints.py \
  --video videos/obra.mp4 \
  --output resultados/obra_anotada.mp4 \
  --report resultados/obra_relatorio.json \
  --max-frames 100
```

Se o codec padrão `mp4v` não estiver disponível, informe um codec aceito pelo
sistema, por exemplo `--codec avc1`.

Os limiares padrão foram calibrados como ponto de partida para esse modelo:
`0.35` para capacetes, `0.55` para pessoas, `0.30` para keypoints e `0.75` para
supressão de pessoas duplicadas. Ajuste-os com vídeos reais do ambiente quando
for necessário priorizar mais detecções ou menos falsos alertas.

Por padrão, cada inferência é organizada em uma pasta própria dentro de `outputs/`:

```text
outputs/
└── image/
    ├── annotated/
    │   └── image_ppe_keypoints.jpg   # imagem ou vídeo anotado
    └── reports/
        └── image_ppe_keypoints.json  # relatório de conformidade
```

### Anotações

- `P1 OK` verde: pessoa com capacete validado na região da cabeça.
- `P1 ?` laranja: track novo ou estado ainda inconclusivo.
- `P1 ALERTA` vermelho: pessoa sem capacete validado.
- `helmet OK` verde: capacete em posição compatível.
- `helmet X` vermelho: capacete fora da região esperada ou sem associação.

Em vídeos, o número após `P` é o ID persistente do tracker, e não a posição da
pessoa na lista de detecções daquele quadro. Pessoas completamente invisíveis não
recebem caixas congeladas; o ID e o último estado continuam guardados apenas para
uma possível reassociação dentro do buffer configurado.

Os keypoints são usados internamente e ficam ocultos por padrão para manter a imagem limpa.

### Resultados das inferências em imagem

As saídas abaixo foram geradas pela pipeline com as cinco imagens de exemplo.

| Image 1 | Image 2 |
| --- | --- |
| ![Inferência da image 1](docs/examples/images/image1_ppe_keypoints.jpg) | ![Inferência da image 2](docs/examples/images/image2_ppe_keypoints.jpg) |

| Image 3 | Image 4 |
| --- | --- |
| ![Inferência da image 3](docs/examples/images/image3_ppe_keypoints.jpg) | ![Inferência da image 4](docs/examples/images/image4_ppe_keypoints.jpg) |

| Image 5 |
| --- |
| ![Inferência da image 5](docs/examples/images/image5_ppe_keypoints.jpg) |

### Melhores momentos dos testes em vídeo

Os exemplos foram selecionados entre cinco inferências em vídeo, priorizando
quadros com detecções consistentes e capacetes claramente visíveis. Caixas verdes
indicam capacete validado; caixas vermelhas representam alerta.

**Plano aberto — quadro 304 (12,67 s).** Quatro pessoas detectadas e três
capacetes validados. É o melhor exemplo para avaliar a associação em um cenário
com várias pessoas.

![Plano aberto: três capacetes validados](docs/examples/video_13261762_momento.gif)

**Plano próximo — quadro 400 (8,00 s).** Duas pessoas e dois capacetes
validados; é o exemplo mais claro para inspeção visual da região da cabeça.

![Plano próximo: dois capacetes validados](docs/examples/video_13751987_momento.gif)

### Opções

```bash
uv run python infer_ppe_keypoints.py \
  --image images/image2.jpg \
  --ppe-checkpoint output_helmet_only_576/checkpoint_best_total.pth \
  --output resultados/imagem_anotada.jpg \
  --report resultados/relatorio.json \
  --ppe-threshold 0.35 \
  --keypoint-threshold 0.55
```

| Opção | Descrição |
| --- | --- |
| `--image` | Imagem de entrada; obrigatória quando `--video` não for usado. |
| `--video` | Vídeo de entrada; mutuamente exclusivo com `--image`. |
| `--ppe-checkpoint` | Checkpoint do detector de EPIs. |
| `--hf-repo-id` | Repositório do Hub usado quando o checkpoint padrão está ausente. |
| `--output` | Caminho da imagem ou vídeo anotado. |
| `--report` | Caminho do relatório JSON. |
| `--ppe-threshold` | Limiar de capacete; padrão `0.35`. |
| `--keypoint-threshold` | Limiar de pessoas; padrão `0.55`. |
| `--keypoint-confidence` | Confiança mínima de keypoint; padrão `0.30`. |
| `--person-nms-iou` | IoU máximo antes de suprimir pessoas duplicadas; padrão `0.75`. |
| `--draw-keypoints` | Exibe keypoints para depuração. |
| `--hide-person-boxes` | Oculta as caixas de pessoas. |
| `--codec` | Codec de quatro caracteres para saída de vídeo; padrão `mp4v`. |
| `--max-frames` | Limita a quantidade de quadros processados (teste). |
| `--track-buffer-seconds` | Retém uma identidade oclusa; padrão `1.0` segundo. |
| `--helmet-window-seconds` | Janela de confirmação da ausência; padrão `1.0` segundo. |

## Relatório JSON

O relatório registra a validação por pessoa:

```json
{
  "pessoas": [
    {
      "id": 1,
      "epis_validados": 1,
      "epis": {
        "capacete": {
          "status": "validado",
          "status_instantaneo": "ausente",
          "temporalmente_retido": true
        }
      }
    }
  ]
}
```

`status_instantaneo` registra a observação do quadro, enquanto `status` é o valor
estabilizado usado na anotação e em `epis_validados`. Quando os dois divergem,
`temporalmente_retido` é `true`: uma falha isolada ou uma observação inconclusiva
não alterou imediatamente o último estado confiável. Um novo track começa como
`nao_verificavel`; `validado` recupera imediatamente o estado protegido e
`ausente` exige mais da metade da janela configurada.

Estados possíveis: `validado`, `fora_da_regiao`, `nao_verificavel` e `ausente`.

## Estrutura

```text
.
├── .github/workflows/               # testes automatizados do repositório
├── configs/                         # receitas YAML de treinamento
│   ├── config_helmet_smoke_576.yaml # teste de VRAM em 576 px
│   └── config_helmet_only_576.yaml  # treino final focado em capacete
├── docs/
│   ├── examples/                    # resultados pequenos selecionados
│   └── PUBLISHING.md                # política GitHub × Hugging Face
├── huggingface/                     # fontes da model card e dos metadados do Hub
├── images/                          # imagens leves de demonstração
├── scripts/publish_hf.py            # valida e publica uma release fechada no Hub
├── tests/                           # testes unitários sem download de pesos
├── models/                          # pesos locais de inferência (ignorado pelo Git)
│   └── helmet_only_576_best.pth
├── construction-ppe/                # dataset original, ignorado pelo Git
├── construction-ppe-helmet/         # dataset derivado de uma classe, ignorado pelo Git
├── videos/                          # entradas locais de vídeo (ignorado pelo Git)
├── outputs/                         # resultados completos (ignorado pelo Git)
├── infer_ppe_keypoints.py           # inferência e validação de capacete
├── evaluate_helmet.py                # avaliação COCO do checkpoint no split de teste
├── model_hub.py                     # resolução local e download automático do modelo
├── prepare_helmet_dataset.py         # filtragem e normalização YOLO para helmet
├── train.py                         # treinamento do RFDETRSmall
├── pyproject.toml                   # dependências e metadados do projeto
└── uv.lock                          # versões bloqueadas das dependências
```

Os arquivos `*_ppe_keypoints.json` e imagens anotadas gerados durante a inferência
são artefatos locais; escolha explicitamente quais exemplos deseja versionar.

## Publicação do modelo

O manifesto do Hugging Face é validado contra nome, tamanho e SHA-256 antes do
upload. Para auditar sem publicar:

```bash
uv run python scripts/publish_hf.py --dry-run
```

Depois de atualizar checkpoint, métricas e model card, publique com:

```bash
hf auth login
uv run python scripts/publish_hf.py
```

Consulte [a política de publicação](docs/PUBLISHING.md) para a lista exata do que
pertence ao GitHub, ao Hugging Face e somente ao ambiente local.

## Limitações

- A regra atual avalia exclusivamente capacete.
- Pessoas pequenas, oclusas ou com keypoints imprecisos podem gerar resultado inconclusivo.
- O tracking usa movimento e sobreposição de caixas, sem reconhecimento facial;
  cruzamentos muito longos ou oclusões maiores que o buffer ainda podem trocar IDs.
- Calibre os limiares com imagens reais do ambiente de operação.
- Resultados devem ser revisados antes de decisões críticas de segurança.
