# train_and_dump_model.py

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. Configuração Inicial ---
# Garante que o diretório para salvar os artefatos exista
os.makedirs("pickle", exist_ok=True)
print("Diretório 'pickle' criado/verificado.")

# --- 2. Carga dos Dados ---
# Carrega o dataset. Certifique-se que o arquivo "incendios.csv" está na mesma pasta.
try:
    df = pd.read_csv(R"C:\Users\nicko\Desktop\FIAP\GS\Front-GS1\CP2-Front\incendios.csv")
    print(f"Dataset 'incendios.csv' carregado com sucesso. Shape inicial: {df.shape}")
except FileNotFoundError:
    print("Erro: Arquivo 'incendios.csv' não encontrado. Abortando.")
    exit()

# --- 3. Pré-processamento ---
print("Iniciando pré-processamento dos dados...")

# 3.1. Remover colunas de identificação e geográficas não utilizadas
df = df.drop(['ID', 'latitude', 'longitude'], axis=1, errors='ignore')
print("Colunas 'ID', 'latitude', 'longitude' removidas.")

# 3.2. Processar a coluna 'data_hora' para extrair o horário
df['data_hora'] = pd.to_datetime(df['data_hora'], errors='coerce')
df['horario'] = df['data_hora'].dt.hour
df = df.drop('data_hora', axis=1, errors='ignore')
df.dropna(subset=['horario'], inplace=True) # Remove linhas se a conversão de data falhou
df['horario'] = df['horario'].astype(int)
print("Coluna 'horario' extraída de 'data_hora'.")

# 3.3. Aplicar One-Hot Encoding para 'tipo_uso_solo'
df = pd.get_dummies(df, columns=['tipo_uso_solo'], drop_first=True, dtype=int)
print("Variáveis dummy criadas para 'tipo_uso_solo'.")

# 3.4. Remover outliers nas colunas numéricas principais usando o método IQR
# É uma boa prática fazer isso antes de escalar os dados
continuous_cols = ['temperatura_c', 'umidade_percentual', 'velocidade_vento_kmh', 'intensidade_fogo_frp']
for col in continuous_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lim_inf = Q1 - 1.5 * IQR
    lim_sup = Q3 + 1.5 * IQR
    
    initial_rows = len(df)
    df = df[(df[col] >= lim_inf) & (df[col] <= lim_sup)]
    print(f"Removidos {initial_rows - len(df)} outliers da coluna '{col}'.")

# 3.5. Garantir que não há NaNs restantes
df.dropna(inplace=True)
print(f"Shape final após limpeza e pré-processamento: {df.shape}")


# --- 4. Separação de Features (X) e Alvo (y) ---
y = df['intensidade_fogo_frp']
X = df.drop('intensidade_fogo_frp', axis=1)

# --- 5. Salvar a Lista de Colunas do Modelo ---
# Essencial para garantir que a ordem e o número de colunas sejam os mesmos na predição
colunas_modelo = list(X.columns)
joblib.dump(colunas_modelo, "pickle/colunas_modelo.pkl")
print("Lista de colunas do modelo salva em 'pickle/colunas_modelo.pkl'.")


# --- 6. Divisão em Dados de Treino e Teste ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Dados divididos em treino ({len(X_train)} linhas) e teste ({len(X_test)} linhas).")

# --- 7. Escalonamento das Features ---
# O escalonamento é crucial para modelos lineares.
# Usamos fit_transform no treino e apenas transform no teste.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Salvar o scaler para ser usado na aplicação
joblib.dump(scaler, "pickle/scaler.pkl")
print("Scaler salvo em 'pickle/scaler.pkl'.")

# --- 8. Treinamento do Modelo de Regressão Linear ---
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("Modelo de Regressão Linear treinado.")

# --- 9. Avaliação do Desempenho (Opcional, mas recomendado) ---
y_pred = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- Desempenho do Modelo no Conjunto de Teste ---")
print(f"RMSE (Raiz do Erro Quadrático Médio): {rmse:.4f}")
print(f"R² (Coeficiente de Determinação): {r2:.4f}")
print("-------------------------------------------------")


# --- 10. Salvar o Modelo Treinado ---
joblib.dump(model, "pickle/modelo_regressao_linear.pkl")
print("Modelo salvo em 'pickle/modelo_regressao_linear.pkl'.")

print("\n✅ Processo concluído! Todos os artefatos foram salvos na pasta 'pickle'.")