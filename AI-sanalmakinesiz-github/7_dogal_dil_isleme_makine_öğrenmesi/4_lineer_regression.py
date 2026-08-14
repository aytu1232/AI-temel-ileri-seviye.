"""
pip install scikit-learn matplotlib numpy
"""

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

X = np.random.rand(100,1)
y = 3 + 4*X + np.random.rand(100,1)

lin_reg = LinearRegression()
lin_reg.fit(X,y)

plt.figure()
plt.scatter(X,y)
plt.plot(X,lin_reg.predict(X),color = "red",alpha=0.7)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Lineer regresyon")


a1 = lin_reg.coef_[0][0]
print(f"a1:{a1}")

a0 = lin_reg.intercept_[0]
print(f"a0: {a0}")

for i in range(100):
    y_head = a0 + a1*X
    plt.plot(X,y_head,color = "green",alpha = 0.7)
plt.show()