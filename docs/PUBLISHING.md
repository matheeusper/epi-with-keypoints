# Política de distribuição e publicação

Este projeto usa dois repositórios complementares:

- [GitHub](https://github.com/matheeusper/epi-with-keypoints): código-fonte e
  materiais necessários para reproduzir treino, avaliação e inferência.
- [Hugging Face](https://huggingface.co/matheeusper/epi-with-keypoints): release
  versionada do checkpoint treinado e sua documentação de modelo.

## O que pertence a cada destino

| Conteúdo | GitHub | Hugging Face | Apenas local |
| --- | :---: | :---: | :---: |
| Código Python, testes e CI | ✓ |  |  |
| Configurações e lockfile | ✓ | config final |  |
| Model card fonte e metadados | ✓ | versão publicada |  |
| Checkpoints e pesos |  | ✓ | ✓ |
| Métricas resumidas | ✓ | ✓ |  |
| Imagens e GIFs pequenos selecionados | ✓ |  |  |
| Dataset original ou derivado |  |  | ✓ |
| Vídeos de entrada e saídas completas |  |  | ✓ |
| Caches, ambientes e credenciais |  |  | ✓ |

Os padrões em `.gitignore` impedem que checkpoints, datasets, vídeos, resultados
e segredos locais sejam enviados acidentalmente ao GitHub. O script de publicação
do Hub usa uma lista fechada de quatro arquivos, impedindo o upload do restante do
workspace.

## Manifesto da release no Hugging Face

O repositório de modelo deve conter somente:

```text
README.md
helmet_only_576_best.pth
model_metadata.json
training_config.yaml
```

As fontes desses arquivos no GitHub são:

```text
huggingface/README.md
models/helmet_only_576_best.pth       # local e ignorado pelo Git
huggingface/model_metadata.json
configs/config_helmet_only_576.yaml
```

## Publicação segura

1. Autentique o CLI moderno do Hub:

   ```bash
   hf auth login
   hf auth whoami
   ```

2. Confira nome, tamanho e SHA-256 sem acessar o Hub:

   ```bash
   uv run python scripts/publish_hf.py --dry-run
   ```

3. Publique os quatro artefatos em um único commit:

   ```bash
   uv run python scripts/publish_hf.py
   ```

4. Verifique o conteúdo remoto:

   ```bash
   hf models list matheeusper/epi-with-keypoints --tree --recursive --human-readable
   hf download matheeusper/epi-with-keypoints helmet_only_576_best.pth --dry-run
   ```

Ao substituir o checkpoint, atualize primeiro `huggingface/model_metadata.json`
com o novo nome, tamanho, checksum e métricas. A publicação falha se os valores
não corresponderem ao arquivo local.

## Licenças

O dataset Construction-PPE e o checkpoint derivado estão declarados como
AGPL-3.0. O repositório GitHub ainda não possui uma licença explícita para o código
autoral; essa escolha deve ser feita pelo proprietário antes de aceitar
contribuições ou autorizar redistribuição do código.
