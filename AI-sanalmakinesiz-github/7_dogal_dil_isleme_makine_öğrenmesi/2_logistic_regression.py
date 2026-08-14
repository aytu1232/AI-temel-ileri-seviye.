# from ucimlrepo import fetch_ucirepo
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# import pandas as pd



# heart_disease = fetch_ucirepo(name = "Heart Disease")

# df = pd.DataFrame(data = heart_disease.data.features)

# df["target"] = heart_disease.data.targets

# print(df.head())

# """
#    age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  slope   ca  thal  target
# 0   63    1   1       145   233    1        2      150      0      2.3      3  0.0   6.0       0
# 1   67    1   4       160   286    0        2      108      1      1.5      2  3.0   3.0       2
# 2   67    1   4       120   229    0        2      129      1      2.6      2  2.0   7.0       1
# 3   37    1   3       130   250    0        0      187      0      3.5      3  0.0   3.0       0
# 4   41    0   2       130   204    0        2      172      0      1.4      1  0.0   3.0       0
# """

# if df.isna().any().any():
#     df.dropna(inplace=True)
#     print("nan")


# X = df.drop(["target"], axis= 1).values
# y = df.target.values

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# log_reg = LogisticRegression(penalty="l2",C= 1 ,solver="lbfgs",max_iter=100)
# log_reg.fit(X_train,y_train)

# accuracy = log_reg.score(X_test,y_test)
# print(f"logistic regression accuracy: {accuracy}")

#ödev 0 ve 1 olarak "var yada yok" olarak değiştirip doğruluk oranı yükseltme.

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

# 1. Veri setini yükleme
heart_disease = fetch_ucirepo(name="Heart Disease")

df = pd.DataFrame(data=heart_disease.data.features)
df["target"] = heart_disease.data.targets

# 2. Hedef değişkeni binary (0 ve 1) hâline getirme
# 0 olanlar 0 kalır, 0'dan büyük olanlar (1, 2, 3 vb.) 1 yapılır.
df["target"] = df["target"].apply(lambda x: 0 if x == 0 else 1)

# 3. Eksik verileri temizleme
if df.isna().any().any():
    df.dropna(inplace=True)
    print("Eksik veriler temizlendi.")

# 4. Özellikler (X) ve Hedef (y) ayrımı
X = df.drop(["target"], axis=1).values
y = df.target.values

# 5. Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# 6. Lojistik Regresyon modeli kurma ve eğitme
# Not: Sınıf sayısı 2'ye düştüğü için 'lbfgs' çözücüsü bu ikili sınıflandırma için uygundur.
log_reg = LogisticRegression(penalty="l2", C=1, solver="lbfgs", max_iter=1000)
log_reg.fit(X_train, y_train)

# 7. Doğruluk (accuracy) hesaplama ve yazdırma
accuracy = log_reg.score(X_test, y_test)
print(f"Logistic Regression Accuracy: {accuracy:.4f}")
print(df.head())