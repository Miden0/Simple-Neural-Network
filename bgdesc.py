import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def sigmoid(x_arr):
    output = 1 / (1 + np.exp(-x_arr))
    return output


def log_loss(y_arr, y_predicted_arr):
    epsilon = 1e-15
    yp_new = [min(i, 1-epsilon) for i in y_predicted_arr]
    yp_new = [max(i, epsilon) for i in yp_new]
    yp_new = np.array(yp_new)
    cost = -np.mean(y_arr * np.log(yp_new) + (1 - yp_new) * np.log(1 - yp_new))
    return cost


def gradient_descent(x_df, y_df, epochs):
    x_arr = np.array(x_df)
    y_arr = np.array(y_df)
    learning_rate = 0.5
    w_arr = np.ones(x_arr.shape[1])
    b = 0
    cost = 0
    for i in range(epochs):
        weighted_sum = np.sum(w_arr * x_arr, axis=1) + b  # this is the same as taking the dot product of "w_arr" and "X_arr transposed"
        y_predicted = sigmoid(weighted_sum)
        cost = log_loss(y_arr, y_predicted)

        w_arr_deriv = np.mean(x_arr * (y_predicted - y_arr)[:, np.newaxis], axis=0)  # this basically transposes the array created by "y_predicted - y_arr"
        b_deriv = np.mean(y_predicted - y_arr)

        w_arr = w_arr - learning_rate * w_arr_deriv
        b = b - learning_rate * b_deriv
    return w_arr, b, cost


class Mynn:
    def __init__(self):
        self.W = None
        self.b = None
        self.cost = None

    def fit(self, X, y, epochs):
        self.W, self.b, self.cost = gradient_descent(X, y, epochs)

    def predict(self, X):
        prediction = np.sum(self.W * X, axis=1) + self.b
        prediction = np.round(sigmoid(prediction))
        return prediction


def test():
    df = pd.read_csv("""https://raw.githubusercontent.com
                    /codebasics/deep-learning-keras-tf-tutorial/master/6_gradient_descent/insurance_data.csv""")
    df.age = df.age / 100

    x_train, x_test, y_train, y_test = train_test_split(df[["age", "affordibility"]],
                                                        df["bought_insurance"],
                                                        test_size=0.2, random_state=3)

    coef, intercepts, cost = gradient_descent(x_train, y_train, 5000)

    print(coef)
    print(intercepts)
    print(cost)
