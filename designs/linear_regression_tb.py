import linear_regression
from fxpmath import Fxp

from nmigen import *
from nmigen.cli import main
from nmigen.back.pysim import Simulator, Delay, Settle

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


print("### Creating Linear Regression Model")

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print(f'Coefficients: {regr.coef_}')
# The bias
print(f'Bias: {regr.intercept_}')
# The mean squared error
print(
    f'Mean squared error: {mean_squared_error(diabetes_y_test, diabetes_y_pred)}')
# The coefficient of determination: 1 is perfect prediction
print(
    f'Coefficient of determination: {r2_score(diabetes_y_test, diabetes_y_pred)}')
print()

print("### Building HW Testbench ###")

weights = regr.coef_
bias = regr.intercept_
dut_linear_regression = linear_regression.LinearRegression(
    weights, bias, bit_depth=32)

m = Module()
m.submodules.linear_regression = dut_linear_regression


def process_main():
    x_test_0 = diabetes_X_test[0]
    y_pred_0 = regr.predict(x_test_0.reshape(1, -1))[0]
    x_test_0 = x_test_0.tolist()
    y_pred_0 = float(y_pred_0)
    # print(x_test_0)
    # print(y_pred_0)

    for i, x in enumerate(x_test_0):
        yield dut_linear_regression.x[i].eq(int(Fxp(x, True, 32, 16).base_repr(10)))
    yield Delay(1e-6)
    yield Settle()

    output = yield dut_linear_regression.y
    out_fpx = Fxp(output, True, 32, 16, raw=True)
    # print(out_fpx)
    print("Regression Results")
    print(f"x_sample: {x_test_0}")
    print(f"y_predicted: {y_pred_0}")
    print(f"y_predicted_hw: {out_fpx}")
    print(f'hw_error: {out_fpx-y_pred_0}')
    print()


sim = Simulator(m)
sim.add_process(process_main)
print("Done")
print()

print("### Running Sumliation ###")
sim.run()
print("### Simualtion Complete ###")
