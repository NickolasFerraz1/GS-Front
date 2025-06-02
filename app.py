# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------
# 1) Configuração da página
# -----------------------------------
st.set_page_config(
    page_title='Simulador - Intensidade de Fogo (FRP)',
    page_icon='🔥',
    layout='wide'
)
st.title('Simulador de Intensidade de Fogo (FRP)')

with st.expander('Descrição do App', expanded=False):
    st.markdown("""
        Este simulador utiliza um modelo de regressão linear para prever a **Intensidade de Fogo (FRP)**
        com base em variáveis climáticas e contextuais.
        - **Modo CSV**: faça upload de um arquivo `.csv` com os dados para predição em lote.
        - **Entrada Manual**: preencha manualmente as variáveis para obter uma predição instantânea.
    """)

# -----------------------------------
# 2) Carregamento do modelo, scaler e lista de colunas
# -----------------------------------
@st.cache_resource
def load_model_and_artifacts():
    try:
        model = joblib.load("pickle/modelo_regressao_linear.pkl")
        colunas_modelo = joblib.load("pickle/colunas_modelo.pkl")
        scaler = joblib.load("pickle/scaler.pkl")
        return model, colunas_modelo, scaler
    except FileNotFoundError:
        st.error("Erro: Arquivos do modelo ('pickle/modelo_regressao_linear.pkl', etc.) não encontrados.")
        st.info("Certifique-se de que a pasta 'pickle' com os artefatos do modelo está no mesmo diretório que o app.")
        return None, None, None

model, colunas_modelo, scaler = load_model_and_artifacts()

if not model:
    st.stop() # Interrompe a execução se os artefatos não foram carregados

# Extrair categorias "tipo_uso_solo" a partir das colunas do modelo
tipo_uso_solo_cols = [c for c in colunas_modelo if c.startswith("tipo_uso_solo_")]
tipos_uso_solo = sorted([c.replace("tipo_uso_solo_", "") for c in tipo_uso_solo_cols])

# Lista de features contínuas que o modelo espera (exceto dummies)
cont_features = sorted([
    c for c in colunas_modelo
    if not c.startswith("tipo_uso_solo_")
])

# -----------------------------------
# 3) Controle de Threshold (sidebar)
# -----------------------------------
threshold = st.sidebar.slider(
    'Definir Threshold de Intensidade (FRP)',
    min_value=0.0, max_value=500.0, # Ajustado para a escala de FRP
    value=100.0, step=10.0,
    help="Qualquer valor de FRP previsto ACIMA deste threshold será classificado como 'Alta Intensidade'."
)
st.sidebar.info(f"Threshold atual: **{threshold:.2f} FRP**. Valores acima são 'Alta Intensidade'.")


# -----------------------------------
# 4) Abas: Modo CSV e Entrada Manual
# -----------------------------------
tab_csv, tab_manual = st.tabs(["Modo CSV", "Entrada Manual"])

# --- MODO CSV ---
with tab_csv:
    st.subheader("Upload de CSV para Previsão em Lote")
    st.markdown(f"""
        Envie um arquivo `.csv` contendo as colunas que o modelo espera.
        - **Colunas necessárias**: `{', '.join(colunas_modelo)}`
        - O app vai reordenar as colunas e preencher com zeros as que estiverem faltando.
    """)
    uploaded_file = st.file_uploader("Faça upload do seu CSV", type="csv")

    if uploaded_file is not None:
        try:
            Xtest_raw = pd.read_csv(uploaded_file)
            st.info(f"O CSV foi carregado com {Xtest_raw.shape[0]} linhas e {Xtest_raw.shape[1]} colunas.")

            # 1) Reindexar para manter exatas colunas que o modelo espera
            Xtest = Xtest_raw.reindex(columns=colunas_modelo, fill_value=0)

            # 2) Aplicar scaler + predição
            Xtest_scaled = scaler.transform(Xtest)
            y_scores = model.predict(Xtest_scaled)
            y_pred_binary = (y_scores >= threshold).astype(int)

            # 4) Construir DataFrame de saída
            df_pred = Xtest_raw.copy()
            df_pred["intensidade_prevista_frp"] = y_scores
            df_pred["classificacao_intensidade"] = np.where(y_pred_binary == 1, 'Alta Intensidade', 'Baixa Intensidade')

            st.subheader("Resultados das Previsões")
            with st.expander("Visualizar tabela de predições", expanded=True):
                c1, c2 = st.columns(2)
                alta_intensidade_count = (df_pred["classificacao_intensidade"] == 'Alta Intensidade').sum()
                c1.metric("Total de 'Alta Intensidade'", int(alta_intensidade_count))
                c2.metric("Total de 'Baixa Intensidade'", int(len(df_pred) - alta_intensidade_count))
                
                def highlight_intensity(series):
                    return ['background-color: #ff7f7f' if val == 'Alta Intensidade' else 'background-color: #90ee90' for val in series]

                st.dataframe(df_pred.style.apply(
                    highlight_intensity,
                    subset=['classificacao_intensidade']
                ))

            # Seção de Análise Gráfica
            st.subheader("Análise Gráfica Comparativa")
            with st.expander("Visualizar gráficos", expanded=False):
                st.markdown("""
                    Selecione duas variáveis numéricas do seu arquivo para visualizar a relação entre elas.
                """)

                # 1. Identificar colunas numéricas do arquivo original para plotagem
                numeric_cols = Xtest_raw.select_dtypes(include=np.number).columns.tolist()

                if len(numeric_cols) < 2:
                    st.warning("São necessárias pelo menos duas colunas numéricas no arquivo para gerar um gráfico de dispersão.")
                else:
                    # 2. Permitir que o usuário escolha as colunas para os eixos X e Y
                    st.markdown("##### Selecione as variáveis para os eixos:")
                    c1_select, c2_select = st.columns(2)
                    with c1_select:
                        selected_col_x = st.selectbox("Variável para o Eixo X:", numeric_cols, index=0)
                    with c2_select:
                        default_y_index = 1 if len(numeric_cols) > 1 else 0
                        selected_col_y = st.selectbox("Variável para o Eixo Y:", numeric_cols, index=default_y_index)

                    # Controles de tamanho e aparência do gráfico
                    st.markdown("##### Ajustes do gráfico:")
                    c1_plot, c2_plot, c3_plot = st.columns(3)
                    with c1_plot:
                        width = st.number_input("Largura", min_value=3, max_value=15, value=7, step=1)
                    with c2_plot:
                        height = st.number_input("Altura", min_value=3, max_value=15, value=5, step=1)
                    with c3_plot:
                        alpha = st.slider("Transparência (alpha)", 0.1, 1.0, 0.7, 0.05)

                    # 3. Criar abas para cada tipo de gráfico
                    tab_scatter, tab_hist = st.tabs(["Gráfico de Dispersão", "Histograma"])
                    plot_size = (width, height) 

                    with tab_scatter:
                        if selected_col_x == selected_col_y:
                            st.warning("Selecione variáveis diferentes para os eixos X e Y para uma melhor visualização.")
                        
                        fig_scatter, ax_scatter = plt.subplots(figsize=plot_size)
                        sns.scatterplot(
                            data=df_pred,
                            x=selected_col_x,
                            y=selected_col_y,
                            hue='classificacao_intensidade',
                            ax=ax_scatter,
                            hue_order=['Baixa Intensidade', 'Alta Intensidade'],
                            alpha=alpha
                        )
                        ax_scatter.set_title(f"Relação entre '{selected_col_x}' e '{selected_col_y}'")
                        st.pyplot(fig_scatter)

                    with tab_hist:
                        # Para o histograma, o usuário escolhe qual das duas variáveis selecionadas quer visualizar
                        st.markdown("---")
                        hist_col_choice = st.radio(
                            "Para o Histograma, qual variável deseja analisar?",
                            (selected_col_x, selected_col_y),
                            horizontal=True,
                        )
                        
                        fig_hist, ax_hist = plt.subplots(figsize=plot_size)
                        sns.histplot(
                            data=df_pred,
                            x=hist_col_choice,
                            hue='classificacao_intensidade',
                            kde=True,
                            stat="density",
                            common_norm=False,
                            hue_order=['Baixa Intensidade', 'Alta Intensidade']
                        )
                        ax_hist.set_title(f"Distribuição de '{hist_col_choice}' por Classe")
                        st.pyplot(fig_hist)

        except Exception as e:
            st.error(f"Erro ao processar o CSV: {e}")
                        
# --- MODO ENTRADA MANUAL ---
with tab_manual:
    st.subheader("Preenchimento Manual de Variáveis")

    # 1) Inputs para as variáveis contínuas
    with st.form("manual_input_form"):
        c1, c2 = st.columns(2)
        with c1:
            temperatura_c = st.slider('Temperatura (°C)', 0.0, 50.0, 30.0, step=0.5)
            umidade_percentual = st.slider('Umidade Relativa (%)', 0.0, 100.0, 40.0, step=1.0)
            velocidade_vento_kmh = st.slider('Velocidade do Vento (km/h)', 0.0, 60.0, 15.0, step=1.0)
        with c2:
            horario = st.slider('Hora do Dia (0–23)', 0, 23, 14, step=1)
            confianca = st.slider('Confiança da Detecção (se aplicável)', 0.0, 1.0, 0.9, step=0.01)
            ocorrencia_fogo = st.selectbox('Ocorrência de Fogo já detectada?', [1, 0], help="Use 1 se já há um foco detectado, 0 caso contrário.")
        
        # 2) Escolha do tipo de uso do solo
        tipo_uso_solo_sel = st.selectbox("Tipo de Uso do Solo", ["---"] + tipos_uso_solo)

        submit_button = st.form_submit_button(label="Realizar Predição Manual")

    if submit_button:
        try:
            # 3) Construir dicionário com todas as colunas zeradas
            data_dict = {col: 0 for col in colunas_modelo}

            # 4) Preencher as contínuas
            data_dict["temperatura_c"] = temperatura_c
            data_dict["umidade_percentual"] = umidade_percentual
            data_dict["velocidade_vento_kmh"] = velocidade_vento_kmh
            data_dict["horario"] = horario
            data_dict["confianca"] = confianca
            data_dict["ocorrencia_fogo"] = ocorrencia_fogo
            
            # 5) Preencher a dummy de tipo_uso_solo (se selecionado)
            if tipo_uso_solo_sel != "---":
                key_tipo_solo = f"tipo_uso_solo_{tipo_uso_solo_sel}"
                if key_tipo_solo in data_dict:
                    data_dict[key_tipo_solo] = 1

            # 6) Converter em DataFrame (única linha) na ordem correta
            df_input = pd.DataFrame([data_dict], columns=colunas_modelo)

            st.write("### Dados enviados ao modelo:")
            st.dataframe(df_input)

            # 7) Aplicar scaler e predição
            df_input_scaled = scaler.transform(df_input)
            score = model.predict(df_input_scaled)[0]
            final_pred = int(score >= threshold)

            # 8) Exibir resultado
            st.write("---")
            st.subheader("Resultado da Predição")
            if final_pred == 1:
                st.error(f"🚨 ALTA INTENSIDADE PREVISTA! (FRP: {score:.2f})")
                st.toast('Risco Elevado Detectado!', icon='🔥')
            else:
                st.success(f"✅ Baixa intensidade prevista. (FRP: {score:.2f})")

            with st.expander("Detalhes da predição"):
                st.write(f"- **Intensidade Prevista (FRP)**: {score:.2f}")
                st.write(f"- **Threshold de Intensidade definido**: {threshold:.2f}")
                st.progress(min(score / (threshold * 2), 1.0)) # Normaliza o progresso

        except Exception as e:
            st.error(f"Erro ao fazer a predição: {e}")