import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime

data=pd.read_csv("Ethereum_Historical_Data.csv")


x = np.array(data["Entry"]).reshape((-1, 1))
y = np.array(data["Open"])

model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

y_pred = model.predict(x)

plt.scatter(x, y)
plt.plot(x, y_pred)
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()

print(model.score(x, y))