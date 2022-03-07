Python 3.7.3 (v3.7.3:ef4ec6ed12, Mar 25 2019, 21:26:53) [MSC v.1916 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import pandas as pd
>>> # trainData
>>> trainData = pd.read_csv('path of your csv dataset')
>>> testData = pd.read_csv('path of your csv dataset')
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> vectorizer = TfidfVectorizer(min_df = 2,
                             max_df = 0.5,
                             sublinear_tf = True,
                             use_idf = True)
>>> train_vectors = vectorizer.fit_transform(trainData['Content'])
>>> test_vectors = vectorizer.transform(testData['Content'])
>>> from sklearn import svm
>>> from sklearn.metrics import classification_report
>>> from sklearn.metrics import confusion_matrix
>>> from sklearn.metrics import accuracy_score
>>> classifier_linear = svm.SVC(kernel='linear')
>>> import time
>>> t0 = time.time()
>>> classifier_linear.fit(train_vectors, trainData['Label'])
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
  kernel='linear', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False)
>>> t1 = time.time()
>>> prediction_linear = classifier_linear.predict(test_vectors)
>>> results = confusion_matrix(testData['Label'], prediction_linear)
>>> print(results)
[[5 3]
 [1 6]]
>>> print('Accuracy Score: ', accuracy_score(testData['Label'], prediction_linear))
Accuracy Score:  0.7333333333333333
>>> t2 = time.time()
>>> time_linear_train = t1-t0
>>> time_linear_predict = t2-t1
>>> print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
Training time: 79.964844s; Prediction time: 131.986328s
>>> report = classification_report(testData['Label'], prediction_linear, output_dict=True)
>>> print('positive: ', report['pos'])
positive:  {'precision': 0.6666666666666666, 'recall': 0.8571428571428571, 'f1-score': 0.75, 'support': 7}
>>> print('negative: ', report['neg'])
negative:  {'precision': 0.8333333333333334, 'recall': 0.625, 'f1-score': 0.7142857142857143, 'support': 8}
>>> 
