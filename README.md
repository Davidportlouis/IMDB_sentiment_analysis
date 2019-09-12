# revize-analiz
Machine Learning model to perform sentiment analysis on movie reviews and to predict the whether to watch or avoid an movie based on user reviews 


#dataset 
This dataset was Collected From http://ai.stanford.edu/~amaas/data/sentiment/
citation imdb reviews

#workflow

#Load data into program using load_files
There are 25000 samples in train data composed of 12500 positive and 12500 negative reviews
There are 25000 samples in test data also, composed of 12500 posisitve and 12500 negative reviews

#Representing text data as bag of words using countvectorizer
Countvectorizer transformer converts the input documents into space matrix of features

#Model development using linear_model
Logisitc Regression will be used for this model as for high dimensional spare data, it works best
Grid Search is used for determining best 'C' value in this case {'C' : 0.5}
Cross validation is used to avoid overfitting data

Grid Output (not included in code):
Best cross-validation score: 0.88
Best parameters:  {'C': 1}
Best estimator:  LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='warn',
          n_jobs=None, penalty='l2', random_state=None,  solver='warn',
          tol=0.0001, verbose=0, warm_start=False)

#Custom validation review:
custom reviews are loaded as text file and predicted using the LogisiticRegression Model

#validation accuracy:
'C' : 05.
'accuracy_score' : 0.87
