# Detecção de EPIs com Keypoints usando RF-DETR

Este projeto treina um modelo `RFDETRSmall` para a detecção de Equipamentos de Proteção Individual (EPIs) e seus respectivos keypoints em imagens de canteiros de obras.

O script de treinamento é configurado para ser flexível, utilizando arquivos YAML para gerenciar hiperparâmetros e outras configurações, permitindo a fácil experimentação com diferentes "receitas" (recipes) de treinamento.

## Funcionalidades

- **Modelo:** Utiliza o `RFDETRSmall`, uma variante eficiente do DETR.
- **Configuração Flexível:** Todas as configurações de treinamento são externalizadas em arquivos YAML, facilitando a experimentação.
- **Receitas de Treinamento:** Vem com múltiplas "receitas" pré-configuradas para diferentes cenários (treino rápido, alta qualidade, etc.).
- **Reprodutibilidade:** Garante a reprodutibilidade dos experimentos através da fixação de sementes (seeds).
- **Aumento de Dados:** Inclui augmentations relevantes para o cenário de construção civil, como variações de iluminação e rotação.

## Instalação

1.  **Clone o repositório:**
    ```bash
    git clone <url-do-seu-repositorio>
    cd epi-with-keypoints
    ```

2.  **Crie e ative um ambiente virtual:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # No Windows: .venv\Scripts\activate
    ```

3.  **Instale as dependências:**
    O projeto utiliza `uv` para gerenciamento de pacotes. Se não o tiver, instale-o com `pip install uv`.
    ```bash
    uv pip install -r requirements.txt
    ```
    *Nota: O arquivo `requirements.txt` pode ser gerado a partir do `uv.lock` ou `pyproject.toml` se necessário.*

4.  **Dataset:**
    Este projeto espera que o dataset `construction-ppe` esteja no diretório raiz. Certifique-se de que a estrutura do seu dataset esteja correta.

## Treinamento

O treinamento é iniciado através do script `train.py`, que aceita um arquivo de configuração como argumento.

### Comando Básico

Para iniciar o treinamento, execute o seguinte comando, especificando a receita (arquivo de configuração) que deseja usar:

```bash
python train.py --config <caminho/para/o/arquivo_config.yaml>
```

### Usando as Receitas (Recipes)

Foram preparadas algumas receitas para diferentes cenários de treinamento. Cada uma salvará os resultados em um diretório de saída diferente (`output_base`, `output_fast_train`, etc.).

- **Receita Base (Padrão):**
  ```bash
  python train.py --config config_base.yaml
  ```

- **Treinamento Rápido:** Ideal para verificar se o pipeline está funcionando.
  ```bash
  python train.py --config config_fast_train.yaml
  ```

- **Aumentos de Dados Agressivos:** Focada em criar um modelo mais robusto, pode exigir mais tempo de treinamento.
  ```bash
  python train.py --config config_high_aug.yaml
  ```

- **Learning Rate Baixo:** Para uma convergência mais fina e estável.
  ```bash
  python train.py --config config_low_lr.yaml
  ```

## Estrutura da Configuração

Os arquivos YAML são divididos nas seguintes seções, permitindo que você crie suas próprias receitas facilmente:

- `general`: Configurações gerais como diretórios de saída, caminho do dataset e semente de aleatoriedade.
- `training`: Hiperparâmetros de treinamento como `epochs`, `batch_size` e `learning_rate`.
- `early_stopping`: Parâmetros para a parada antecipada do treinamento, evitando overfitting.
- `augmentations`: Configurações para as técnicas de aumento de dados aplicadas durante o treinamento.
