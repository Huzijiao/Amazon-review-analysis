# Databricks notebook source
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBClassifier

pdf = pd.read_csv('hair_dryer.tsv', sep="	")

# rebalance data
star_count = [pdf[pdf.star_rating == num].star_rating.count() for num in range(1, 6)]
max_star_count = max(star_count)

pdf = pd.concat(
    [pdf[pdf.star_rating == index + 1].sample(frac=max_star_count // star_count[index], replace=True) for index in
     range(5)], ignore_index=True)

# Split data
cv = CountVectorizer(
    max_features=5000,
    encoding="utf-8",
    ngram_range=(1, 3),
    stop_words={"english"})

X = cv.fit_transform(pdf.review_body).toarray()
y = pdf['star_rating']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25)

# Fit model
eval_set = [(X_train, y_train), (X_test, y_test)]
eval_metric = ["auc", "merror"]

model = XGBClassifier()
model.fit(X_train, y_train, eval_set=eval_set, eval_metric=eval_metric, verbose=True)

# make predictions for test data
y_predictions = model.predict(X_test)
y_predictions = [round(value) for value in y_predictions]

# evaluate predictions
accuracy = accuracy_score(y_test, y_predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

print(model.predict(cv.transform([
    "This is the worst product. I will never buy this again!",
    "I love it so much.",
    "Looks fine to me, but too expensive."
]).toarray()))
