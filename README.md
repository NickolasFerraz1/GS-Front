# GS-Front

# Integrantes:

* Nickolas Ferraz - RM558458
* Marcos Paolucci - RM55
* Sandron Oliveira - RM557172


# **Simulador de Intensidade de Incêndios Florestais**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://link-para-seu-app-streamlit.streamlit.app/)

## 📜 Visão Geral do Projeto

Este projeto consiste em uma solução de ponta a ponta para prever a intensidade de incêndios florestais, medida em Potência Radiativa do Fogo (FRP). A solução inclui um modelo de Machine Learning treinado com dados de incêndios e uma aplicação web interativa construída com Streamlit que serve como interface para o modelo.

A ferramenta permite que usuários, como analistas e gestores de recursos, simulem cenários e obtenham predições instantâneas, auxiliando em uma tomada de decisão mais estratégica e baseada em dados.

## ✨ Funcionalidades

* **Modelo Preditivo:** Utiliza Regressão Linear para estimar a intensidade do fogo (FRP).
* **Interface Interativa:** App web limpo e intuitivo construído com Streamlit.
* **Modo de Predição Manual:** Permite ao usuário inserir manualmente variáveis climáticas e contextuais para obter uma predição para um único cenário.
* **Modo de Predição em Lote:** Permite o upload de um arquivo `.csv` com múltiplos cenários para análise em lote.
* **Visualização de Dados:** Geração de gráficos comparativos (Gráfico de Dispersão e Histograma) para analisar os resultados das predições em lote.
* **Threshold Customizável:** Um slider permite ao usuário definir um limiar para classificar a intensidade prevista como "Alta" ou "Baixa".

## 📁 Estrutura de Pastas

```
seu_projeto/
|
|-- pickle/
|   |-- colunas_modelo.pkl
|   |-- modelo_regressao_linear.pkl
|   `-- scaler.pkl
|
|-- app.py                     # Script principal do Streamlit
|-- incendios.csv              # Seu dataset
|-- requirements.txt           # Dependências do Python
|-- gerador_pkl.py             # Script para treinar e salvar o modelo
`-- README.md                  # Este arquivo
```

---

## ⚙️ Como Funciona: O Treinamento do Modelo (`train_and_dump_model.py`)

Este script é responsável por todo o ciclo de vida do modelo de Machine Learning, desde os dados brutos até os artefatos prontos para produção.

1.  **Carga e Limpeza de Dados:** O script inicia carregando o dataset `incendios.csv` e removendo colunas que não serão utilizadas no modelo (como IDs e coordenadas geográficas).

2.  **Pré-processamento e Engenharia de Features:**
    * **Extração de `horario`:** A coluna `data_hora` é processada para extrair apenas a hora do dia, uma variável importante para o modelo.
    * **One-Hot Encoding:** A coluna categórica `tipo_uso_solo` é transformada em colunas numéricas (dummies) para que o modelo possa processá-la.
    * **Remoção de Outliers:** Utiliza o método de Amplitude Interquartil (IQR) para remover valores extremos das principais colunas numéricas, tornando o modelo mais robusto.

3.  **Divisão dos Dados:** O dataset é dividido em um conjunto de features (`X`) e a variável alvo (`y`, que é `intensidade_fogo_frp`).

4.  **Escalonamento (Scaling):** As features são escalonadas usando `StandardScaler`. Este passo é crucial para modelos de regressão, pois garante que todas as variáveis contribuam de forma equilibrada para o resultado, independentemente de suas escalas originais.

5.  **Treinamento do Modelo:** Um modelo de `LinearRegression` é instanciado e treinado com os dados de treino (`X_train_scaled`, `y_train`).

6.  **Salvando os Artefatos:** Ao final, o script salva três arquivos essenciais na pasta `pickle/`:
    * `modelo_regressao_linear.pkl`: O objeto do modelo treinado.
    * `scaler.pkl`: O objeto do `StandardScaler` "ajustado" aos dados de treino. É fundamental para processar novas entradas da mesma forma.
    * `colunas_modelo.pkl`: A lista exata de colunas que o modelo espera receber. Garante que a aplicação web envie os dados na ordem e formato corretos.

---

## 🖥️ Como Funciona: A Aplicação Streamlit (`app.py`)

Este script cria a interface web interativa para o modelo.

1.  **Carregamento dos Artefatos:** No início, o app carrega o modelo, o scaler e a lista de colunas salvos pelo script de treinamento.

2.  **Interface do Usuário:**
    * A tela é dividida em duas abas principais: "Modo CSV" e "Entrada Manual".
    * Uma barra lateral (`sidebar`) contém um slider para que o usuário defina o `threshold` de intensidade.

3.  **Modo de Entrada Manual:**
    * Renderiza sliders e caixas de seleção para que o usuário insira os valores de cada feature (temperatura, umidade, etc.).
    * Ao clicar no botão "Realizar Predição", o app:
        1.  Cria um DataFrame de uma única linha com os dados inseridos.
        2.  Aplica o `scaler` carregado para transformar os dados.
        3.  Usa o `modelo.predict()` para obter a previsão de intensidade (FRP).
        4.  Exibe o resultado de forma clara, com uma animação ou alerta visual.

4.  **Modo CSV:**
    * Permite o upload de um arquivo `.csv`.
    * Após o upload, o script:
        1.  Usa `reindex` para garantir que as colunas do arquivo correspondam exatamente às `colunas_modelo`, preenchendo com zero as que estiverem faltando.
        2.  Aplica o `scaler` e o `modelo.predict()` em todas as linhas do arquivo (predição em lote).
        3.  Exibe os resultados em uma tabela, colorindo as linhas com base na classificação de intensidade.
        4.  Oferece uma seção de **Análise Gráfica**, onde o usuário pode selecionar variáveis para gerar gráficos de dispersão e histogramas, comparando as distribuições entre os grupos de "Alta" e "Baixa" intensidade.

---

## 🚀 Como Executar Localmente e Fazer Deploy na Streamlit Cloud

### Parte 1: Executando o Projeto Localmente

**Pré-requisitos:** Python 3.8+ instalado.

**Passo 1: Crie o arquivo `requirements.txt`**
Crie um arquivo chamado `requirements.txt` na pasta principal do seu projeto e adicione as seguintes bibliotecas:
```txt
pandas
scikit-learn
streamlit
matplotlib
seaborn
```

**Passo 2: Configure o Ambiente Virtual (Recomendado)**
```bash
# Crie o ambiente virtual
python -m venv venv

# Ative o ambiente (no Windows)
.\venv\Scripts\activate

# Ative o ambiente (no macOS/Linux)
source venv/bin/activate
```

**Passo 3: Instale as Dependências**
```bash
pip install -r requirements.txt
```

**Passo 4: Treine o Modelo**
Execute o script de treinamento para gerar os arquivos `.pkl` na pasta `pickle/`.
```bash
python train_and_dump_model.py
```

**Passo 5: Execute o App Streamlit**
```bash
streamlit run app.py
```
Seu navegador abrirá com o aplicativo rodando localmente!

### Parte 2: Deploy na Streamlit Cloud

**Pré-requisitos:** Uma conta no [GitHub](https://github.com/) e uma conta na [Streamlit Cloud](https://share.streamlit.io/signup) (você pode se inscrever com sua conta do GitHub).

**Passo 1: Crie um Repositório no GitHub**
Crie um novo repositório no GitHub e envie **todos** os arquivos do seu projeto para ele:
* A pasta `pickle/` com os artefatos.
* O arquivo `app.py`.
* O arquivo `requirements.txt`.
* Qualquer outra pasta ou arquivo necessário (como a pasta `assets/` se você usar um GIF).

**Passo 2: Acesse a Streamlit Cloud**
Faça login na sua conta da Streamlit Cloud.

**Passo 3: Crie um Novo App**
No seu dashboard, clique no botão **"New app"**.

**Passo 4: Configure o Deploy**
1.  **Repository:** Escolha o repositório do GitHub que você acabou de criar.
2.  **Branch:** Selecione a branch principal (geralmente `main` ou `master`).
3.  **Main file path:** Verifique se o caminho para o arquivo principal é `app.py`.
4.  **App URL:** Customize a URL do seu aplicativo (opcional).

**Passo 5: Faça o Deploy!**
Clique no botão **"Deploy!"**.

A Streamlit Cloud irá automaticamente ler seu arquivo `requirements.txt`, instalar todas as dependências e iniciar sua aplicação. Você pode acompanhar o processo de instalação na janela de logs. Após alguns instantes, seu aplicativo estará no ar e acessível para qualquer pessoa através do link gerado!
