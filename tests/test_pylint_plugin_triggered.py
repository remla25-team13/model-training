from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
X_train = [[1, 2], [3, 4]]
y_train = [0, 1]

model.fit(X_train)  # Should trigger "fit-missing-y"
model.predict(X_train)  # Should trigger "predict-on-training-data"