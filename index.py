import pandas as pd
import numpy as np
import os

# Cores e estilos
class Estilo:
    TITULO = "\033[1;36m"      # Ciano
    SUBTITULO = "\033[1;33m"   # Amarelo
    TEXTO = "\033[1;37m"       # Branco
    SUCESSO = "\033[1;32m"     # Verde
    ALERTA = "\033[1;31m"      # Vermelho
    RESET = "\033[0m"


def linha():
    print(Estilo.TEXTO + "-" * 70 + Estilo.RESET)


def bloco_titulo(texto):
    linha()
    print(Estilo.TITULO + texto.center(70) + Estilo.RESET)
    linha()


# -----------------------------------------------------------------
# 1. Carregamento dos Dados
# -----------------------------------------------------------------
bloco_titulo("RECRUTAMENTO GUIADO POR ÁLGEBRA LINEAR")

print(Estilo.SUBTITULO + "Carregando base de dados..." + Estilo.RESET)
df = pd.read_csv("recruitment_data.csv")
print(Estilo.SUCESSO + "✔ Base carregada com sucesso!" + Estilo.RESET)

linha()
print(Estilo.SUBTITULO + "Primeiras linhas da base:" + Estilo.RESET)
print(df.head())

# -----------------------------------------------------------------
# 2. Preparação dos Dados
# -----------------------------------------------------------------
bloco_titulo("PREPARAÇÃO DOS DADOS")

X = df.drop(columns=["HiringDecision"])
y = df["HiringDecision"]

X_np = X.to_numpy()
y_np = y.to_numpy()

print(Estilo.TEXTO + f"Matriz X: {X_np.shape}  |  Vetor y: {y_np.shape}" + Estilo.RESET)

# -----------------------------------------------------------------
# 3. Cálculo dos Pesos (w)
# -----------------------------------------------------------------
bloco_titulo("CÁLCULO DOS PESOS (w)")

print(Estilo.SUBTITULO + "Calculando pesos usando pseudoinversa..." + Estilo.RESET)
w = np.linalg.pinv(X) @ y

df_pesos = pd.DataFrame({
    "Feature": X.columns,
    "Peso (w)": w
}).sort_values(by="Peso (w)", ascending=False)

print(Estilo.SUCESSO + "✔ Pesos calculados!" + Estilo.RESET)
linha()
print(df_pesos)

# -----------------------------------------------------------------
# 4. Ranking dos Candidatos
# -----------------------------------------------------------------
bloco_titulo("RANKING DOS CANDIDATOS")

scores = X_np @ w
df_scores = df.copy()
df_scores["Score"] = scores
df_scores = df_scores.sort_values(by="Score", ascending=False).reset_index(drop=True)

print(Estilo.SUBTITULO + "TOP 5 MELHORES CANDIDATOS:" + Estilo.RESET)
print(df_scores.head(5))

# -----------------------------------------------------------------
# 5. Novo Candidato
# -----------------------------------------------------------------
bloco_titulo("NOVO CANDIDATO")

novo_candidato = np.array([[
    29, 1, 3, 4, 1, 12, 79, 88, 74, 2
]])

print(Estilo.TEXTO + "Valores do novo candidato:" + Estilo.RESET)
print(novo_candidato)

score_novo = float(novo_candidato @ w)

print(Estilo.SUBTITULO + "\nCalculando score..." + Estilo.RESET)
print(Estilo.SUCESSO + f"✔ Score: {score_novo:.6f}" + Estilo.RESET)

# ranking geral
posicao = (df_scores["Score"] > score_novo).sum() + 1

linha()
print(Estilo.TITULO + "RESULTADO FINAL".center(70) + Estilo.RESET)
linha()

print(Estilo.SUCESSO + f"Score do novo candidato: {score_novo:.6f}" + Estilo.RESET)
print(Estilo.SUCESSO + f"Posição no ranking: {posicao} de {len(df_scores)}" + Estilo.RESET)

linha()
print(Estilo.TITULO + "FIM DA EXECUÇÃO".center(70) + Estilo.RESET)
linha()
