import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("mnist_test.csv")

data.head()

X = data.iloc[:, 1:]
y = data.iloc[:, 0]

X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=20,
    random_state=42
)

mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

sample = X_test.iloc[0].values.reshape(1, -1)

prediction = mlp.predict(sample)

print("Predicted Digit:", prediction[0])

import pickle

pickle.dump(mlp, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))