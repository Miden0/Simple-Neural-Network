from bgdesc import Mynn
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/codebasics/deep-learning-keras-tf-tutorial/master/6_gradient_descent/insurance_data.csv")
df.age = df.age / 100


x_train, x_test, y_train, y_test = train_test_split(df[["age", "affordibility"]],
                                                    df["bought_insurance"], test_size=0.2, random_state=3)



model = Mynn()

model.fit(x_train, y_train, 5000)
print(model.predict(x_test))
print(y_test)
print(f"cost: {model.cost}")