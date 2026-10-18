import os

import numpy as np
from sklearn.metrics import r2_score

MODEL_PATH = env_var = os.environ["MODEL_PATH"]

np.random.seed(9)

#edad
x = np.random.uniform(0,20,100)

#gasto
tendencia = 500 - 145 * x + 14 * (x**2)
ruido = np.random.normal(0, 40, 100)
y = tendencia + ruido

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

mymodel = np.poly1d(np.polyfit(train_x, train_y, 3))
print("Model trained successfully")

r2 = r2_score(test_y, mymodel(test_x))

print("Model Score:", r2)

np.save(MODEL_PATH, mymodel)
