import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('assignment2_dataset_cars.csv')
y = data['price']
x = data.drop('price', axis=1, inplace=False)
print(data.isnull().sum())
print(y)
print(x)
print(data)
plt.scatter(x['year'], y)
plt.xlabel('year', fontsize = 20)
plt.ylabel('price', fontsize = 20)
# plt.plot(X1, prediction, color='red', linewidth = 3)
# plt.show()