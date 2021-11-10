import numpy as np
from sklearn import linear_model, datasets

diabetes = datasets.load_diabetes()
print(type(diabetes), diabetes.data.shape, diabetes.target.shape)
dataForTest = -60

data_train = diabetes.data[:dataForTest]
target_train = diabetes.target[:dataForTest]
print("data trained:", data_train.shape)
print("target trained:", target_train.shape)
data_test = diabetes.data[dataForTest:]
target_test = diabetes.target[dataForTest:]
print("data test:", data_test.shape)
print("target test:", target_test.shape)
