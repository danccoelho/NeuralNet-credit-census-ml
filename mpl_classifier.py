# %%
# Importação das bibliotecas necessárias para o projeto.
# pandas: Usado para manipulação e análise de dados tabulares (DataFrames).
# numpy: Fundamental para operações numéricas, especialmente com arrays.
# pickle: Utilizado para serializar e desserializar objetos Python, neste caso, para carregar os datasets pré-processados.
# sklearn.neural_network.MLPClassifier: A classe principal para construir a Rede Neural Perceptron Multicamadas.
# sklearn.metrics.accuracy_score: Para calcular a acurácia do modelo.
# sklearn.metrics.classification_report: Gera um relatório detalhado com métricas como precisão, recall e F1-score.
# yellowbrick.classifier.ConfusionMatrix: Biblioteca para visualizar a matriz de confusão.
import pandas as pd
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt

# %%
# Carregamento do dataset de crédito pré-processado.
# O arquivo 'credit.pkl' contém os dados já divididos em conjuntos de treinamento e teste.
# As variáveis são:
# - X_credit_treinamento: Features de treinamento para o modelo de crédito.
# - y_credit_treinamento: Variável alvo de treinamento para o modelo de crédito.
# - X_credit_teste: Features de teste para o modelo de crédito.
# - y_credit_teste: Variável alvo de teste para o modelo de crédito.
with open('credit.pkl', 'rb') as f:
    X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)

# %%
# Verificação das dimensões (shape) dos conjuntos de treinamento do dataset de crédito.
# Isso confirma o número de amostras (linhas) e features (colunas) para X, e o número de amostras para y.
X_credit_treinamento.shape, y_credit_treinamento.shape

# %%
# Verificação das dimensões (shape) dos conjuntos de teste do dataset de crédito.
# Essencial para garantir que os dados estejam consistentes para a avaliação do modelo.
X_credit_teste.shape, y_credit_teste.shape

# %%
# Inicialização e treinamento do modelo MLPClassifier para o dataset de crédito.
# max_iter: Número máximo de iterações (épocas) para o otimizador.
# verbose: Define se o progresso do treinamento será impresso.
# tol: Tolerância para a otimização (critério de parada).
# solver: Algoritmo de otimização dos pesos ('adam' é um otimizador popular).
# activation: Função de ativação das camadas ocultas ('relu' é comum para MLPs).
# hidden_layer_sizes: Tupla definindo a arquitetura das camadas ocultas (neste caso, duas camadas com 2 neurônios cada).
rede_neural_credit = MLPClassifier(max_iter=1500, verbose=True, tol=0.0000100, solver='adam', activation='relu', hidden_layer_sizes=(2,2))
rede_neural_credit.fit(X_credit_treinamento, y_credit_treinamento)

# %%
# Realização de previsões no conjunto de teste do dataset de crédito.
# 'previsores_credit' conterá as classes previstas pelo modelo para os dados de teste.
previsores = rede_neural_credit.predict(X_credit_teste)
previsores

# %%
# Exibição dos valores reais da variável alvo do conjunto de teste de crédito.
# Útil para comparar visualmente com as previsões (previsores_credit).
y_credit_teste

# %%
# Cálculo e exibição da acurácia do modelo para o dataset de crédito.
# A acurácia mede a proporção de previsões corretas sobre o total de previsões.
accuracy_score(y_credit_teste, previsores)

# %%
# Geração e exibição da Matriz de Confusão para o modelo de crédito.
# A matriz de confusão visualiza o desempenho do algoritmo de classificação.
# Os eixos mostram True Positives, True Negatives, False Positives e False Negatives.
cm = ConfusionMatrix(rede_neural_credit)
cm.fit(X_credit_treinamento, y_credit_treinamento)
cm.score(X_credit_teste, y_credit_teste)
# Salvar a matrix como imagem
plt.savefig('confusion_matrix_credit.png')

# %%
# Exibição do relatório de classificação detalhado para o modelo de crédito.
# Inclui precisão, recall, F1-score e suporte para cada classe.
print(classification_report(y_credit_teste,previsores))

# %%
# Carregamento do dataset de censo pré-processado.
# Reutiliza os nomes das variáveis, o que pode levar a confusões se os datasets tiverem diferentes tamanhos iniciais ou processamento.
with open('census.pkl', 'rb') as f:
    X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)

# %%
# Inicialização e treinamento do modelo MLPClassifier para o dataset de censo.
# Nota: hidden_layer_sizes foi ajustado para (55,55) neste modelo, indicando uma arquitetura diferente.
rede_neural_census = MLPClassifier(max_iter=1000, verbose=True, tol=0.000010, hidden_layer_sizes=(55,55))
rede_neural_census.fit(X_credit_treinamento, y_credit_treinamento)

# %%
# Realização de previsões no conjunto de teste do dataset de censo.
previsores = rede_neural_census.predict(X_credit_teste)
previsores

# %%
# Exibição dos valores reais da variável alvo do conjunto de teste de censo.
y_credit_teste

# %%
# Cálculo e exibição da acurácia do modelo para o dataset de censo.
accuracy_score(y_credit_teste,previsores)

# %%
# Geração e exibição da Matriz de Confusão para o modelo de censo.
cm = ConfusionMatrix(rede_neural_census)
cm.fit(X_credit_treinamento, y_credit_treinamento)
cm.score(X_credit_teste, y_credit_teste)
# Salvar a matrix como imagem
plt.savefig('confusion_matrix_census.png')

# %%
# Exibição do relatório de classificação detalhado para o modelo de censo.
print(classification_report(y_credit_teste,previsores))


