import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sktime.performance_metrics.forecasting import mean_absolute_error
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.model_selection import temporal_train_test_split

# Dados do arquivo CSV
data = pd.read_csv('app/utils/dados_historicos_Ibovespa.csv', delimiter=',', decimal=',', parse_dates=['Data'], dayfirst=True)

# Selecionar a coluna último e converter para valores numéricos
y = pd.to_numeric(data['Último'], errors='coerce')

# Remover valores NaN resultantes da conversão
y = y.dropna()

# Dividir os dados em conjuntos de treino e teste
y_train, y_test = temporal_train_test_split(y, test_size=0.2)

# Função para testar diferentes estratégias do Naive Forecaster
def test_naive_forecaster(strategy):
    forecaster = NaiveForecaster(strategy=strategy)
    forecaster.fit(y_train)
    fh = list(range(1, len(y_test) + 1))
    y_pred = forecaster.predict(fh)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'Strategy: {strategy}, Mean Absolute Error: {mae}')

    # Plotar os resultados
    plt.figure(figsize=(10, 6))
    plt.plot(y_train.index, y_train, label='Train')
    plt.plot(y_test.index, y_test, label='Test')
    plt.plot(y_test.index, y_pred, label=f'Predicted ({strategy})')
    plt.title(f'Naive Forecaster - Strategy: {strategy}')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

# Testar diferentes estratégias
strategies = ["last", "mean", "drift"]
for strategy in strategies:
    test_naive_forecaster(strategy)

# Criar e treinar o modelo Naive Forecaster
forecaster = NaiveForecaster(strategy="last")
forecaster.fit(y_train)

# Fazer previsões
fh = list(range(1, len(y_test) + 1))
y_pred = forecaster.predict(fh)

# Avaliar o modelo
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')