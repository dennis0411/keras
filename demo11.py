from sklearn import datasets

data1 = datasets.make_regression(10, 6, noise=5)
regressionX = data1[0]
print(regressionX)
r1 = sorted(regressionX, key=lambda t: t[0])
print(r1)
r2 = sorted(regressionX, key=lambda t: t[1])
print(r2)
r3 = sorted(regressionX, key=lambda t: t[2])
print(r3)
r4 = sorted(regressionX, key=lambda t: t[3])
print(r4)
r5 = sorted(regressionX, key=lambda t: t[4])
print(r5)
r6 = sorted(regressionX, key=lambda t: t[5])
print(r6)