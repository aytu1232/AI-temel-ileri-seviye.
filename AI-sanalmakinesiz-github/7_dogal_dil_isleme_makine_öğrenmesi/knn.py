"""
pip install scikit-learn pandas matplotlib
"""

from sklearn.datasets import load_breast_cancer #veri seti
from sklearn.neighbors import KNeighborsClassifier # sınıflandırıcı
from sklearn.metrics import accuracy_score , confusion_matrix #değerlendir metrikleri
from sklearn.model_selection import train_test_split # train test ayrımı icin gerekli
from sklearn.preprocessing import StandardScaler #normalizasyon

import pandas as pd
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
df = pd.DataFrame (data = cancer.data , columns= cancer.feature_names)
df["target"] = cancer.target
print(df.head())

X = cancer.data
Y = cancer.target

x_train, x_test,y_train,y_test = train_test_split(X, Y, test_size =0.3,random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

knn = KNeighborsClassifier(n_neighbors= 3)

knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)

print("accuracy : {accuracy}".format(accuracy=accuracy))

conf_matrix = confusion_matrix(y_test,y_pred)

print("confusion matrix \n {conf_matrix}".format(conf_matrix=conf_matrix))

"""
[5 rows x 31 columns]
accuracy : 0.9590643274853801
confusion matrix 
 [[ 59   4]
 [  3 105]]
 """

accuracy_values = []
k_values = []

for k in range(1,21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    y_pred = knn.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    accuracy_values.append(accuracy)

    k_values.append(k)

# for k in range(1, 21):
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(x_train, y_train)
#     y_pred = knn.predict(x_test)
#     accuracy = accuracy_score(y_test, y_pred)
    
#     accuracy_values.append(accuracy)
#     k_values.append(k)  # Only append k here!

plt.figure()
plt.plot(k_values,accuracy_values,marker = "o",linestyle = "-")
plt.title("k değerine karşılık gelen accuracy")
plt.xlabel("k değeri")
plt.ylabel("doğruluk")
plt.show()