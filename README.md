# Zhalgas-Practice-12
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Генерация примерных данных
np.random.seed(42)
budget = np.random.randint(1000, 5000, 100)
channels = np.random.randint(1, 5, 100)
seasonality = np.random.normal(0, 1, 100)
ad_costs = 200 * budget + 300 * channels + 50 * seasonality + np.random.normal(0, 500, 100)

# Создание DataFrame из данных
import pandas as pd
data = pd.DataFrame({'Budget': budget, 'Channels': channels, 'Seasonality': seasonality, 'AdCosts': ad_costs})

# Разделение данных на обучающий и тестовый наборы
X = data[['Budget', 'Channels', 'Seasonality']]
y = data['AdCosts']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Построение линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Предсказание на тестовом наборе данных
y_pred = model.predict(X_test)

# Оценка точности модели
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Визуализация результатов
plt.scatter(X_test['Budget'], y_test, label='Фактические затраты на рекламу')
plt.scatter(X_test['Budget'], y_pred, label='Предсказанные затраты на рекламу', marker='o')
plt.xlabel('Бюджет')
plt.ylabel('Затраты на рекламу')
plt.legend()
plt.show()




def calculate_speed(time1, distance1, time2, distance2):
    # Рассчитываем скорость как отношение изменения расстояния к изменению времени
    speed = (distance2 - distance1) / (time2 - time1)
    return speed

time1 = 0  # Время в часах
distance1 = 0  # Расстояние в километрах

time2 = 2  # Время в часах
distance2 = 150  # Расстояние в километрах

result_speed = calculate_speed(time1, distance1, time2, distance2)
print(f"Скорость автомобиля: {result_speed} км/ч")
