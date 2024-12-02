# Sistema de Recomendação de Livros

Este projeto implementa um sistema de recomendação de livros baseado em características dos livros, como autor, idioma e editora. O sistema utiliza **TF-IDF** e **similaridade cosseno** para gerar recomendações personalizadas.

## Pré-requisitos

Antes de executar o projeto, é necessário instalar as dependências. Utilize o arquivo `requirements.txt` para instalar as bibliotecas necessárias.

### Instalar Dependências

1. Clone este repositório ou faça o download do projeto.
2. Navegue até a pasta do projeto e execute o seguinte comando para instalar as dependências:

```bash
pip install -r requirements.txt
```

## Arquivo de Dados

O **dataset** está localizado na pasta raiz e se chama `books.csv`. Este arquivo contém informações sobre os livros, como `bookID`, `title`, `authors`, `language_code`, `publisher`, entre outros.

## Como Executar

### 1. Visualizar Pré-Processamento

Para visualizar o pré-processamento e o carregamento dos dados, execute o seguinte comando:

```bash
python preview.py
```

### 2. Gerar Recomendações

Para gerar recomendações de livros com base em um `bookID` específico, execute:

```bash
python recommendations.py
```

### 3. Avaliar o Desempenho

Para avaliar o desempenho do sistema de recomendação utilizando métricas como precisão, revocação e F1-score, execute:

```bash
python evaluate.py
```

## Estrutura de Arquivos

* `books.csv`: O dataset contendo os dados dos livros.
* `preview.py`: Código para visualização e pré-processamento dos dados.
* `recommendations.py`: Algoritmo para gerar recomendações com base em características dos livros.
* `evaluate.py`: Código para avaliar a performance do sistema de recomendação.

## Dependências

Este projeto utiliza as seguintes bibliotecas Python:

* `pandas`
* `scikit-learn`
* `numpy`

Você pode verificar todas as dependências no arquivo `requirements.txt`.
