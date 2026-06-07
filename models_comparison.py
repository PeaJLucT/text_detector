import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ссылка на файл с результатами каждой модели на фотографиях
df = pd.read_csv("model_results.csv")

# Строим график
plt.figure(figsize=(10, 6))
# ВЫвод графика c средним значениям найденных слов по каждой модели за 20 изображений
sns.barplot(data=df, x="model", y="number of found", palette="viridis")
plt.title("Среднее количество найденных объектов по моделям")
plt.xlabel("Название модели")
plt.ylabel("Среднее кол-во")
plt.show()

#вывод графика с статистикой каждого изображения и моделей на них
# plt.figure(figsize=(14, 7))
# sns.barplot(data=df, x="image", y="number of found", hue="model")

# plt.title("Сравнение моделей для каждого изображения")
# plt.xticks(rotation=45, ha='right') 
# plt.ylabel("Количество находок")
# plt.xlabel("Имя изображения")
# plt.tight_layout()
# plt.show()