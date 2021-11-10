from sklearn import linear_model

features = [[0, 1], [1, 3], [2, 8]]
values = [1, 4, 5.5]
regression1 = linear_model.LinearRegression()
regression1.fit(features, values)

print("coef", regression1.coef_)
print("intercept", regression1.intercept_)
print("score", regression1.score(features, values))
new_features = [[0, 0], [2, 2], [4, 4], [8, 8]]
guess = regression1.predict(new_features)
print("predict", guess)
print("score2:", regression1.score(new_features, guess))