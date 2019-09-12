import numpy as np
import matplotlib.pyplot as plt
import mglearn
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

train_data = load_files('aclImdb/train')
text_train, y_train = train_data.data, train_data.target

test_data = load_files('aclImdb/test')
text_test, y_val = test_data.data, test_data.target

print("Running Model")

cv = CountVectorizer(min_df=5, ngram_range=(2, 2))
X_train = cv.fit(text_train).transform(text_train)
x_val = cv.transform(text_test)

feature_names = cv.get_feature_names()

model = LogisticRegression(C=0.5, class_weight=None, dual=False, fit_intercept=True,
                           intercept_scaling=1, l1_ratio=None, max_iter=100,
                           multi_class='warn', n_jobs=None, penalty='l2',
                           random_state=None, solver='warn', tol=0.0001, verbose=0,
                           warm_start=False)
model.fit(X_train, y_train)
model.predict(x_val)
print(f"Validation Score: {model.score(x_val, y_val)}")


with open('validation.txt', 'r') as test:

    predict = model.predict(cv.transform(test))

print(predict)

if predict.all() == 1:
    print("This Movie is Good to Watch")
else:
    print("It is best to avoid this movie")
