Python 3.7.3 (v3.7.3:ef4ec6ed12, Mar 25 2019, 21:26:53) [MSC v.1916 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> # importimi i paketes per marrjen e dataseteve
>>> import pandas as pd
>>> # trainData merr datasetin per trajnim
>>> trainData = pd.read_csv('C:/Users/Admin/Desktop/Tema e Diplomes/KomentetTrain.csv')
>>> # trainData merr datasetin per testim
>>> testData = pd.read_csv('C:/Users/Admin/Desktop/Tema e Diplomes/KomentetTest.csv')
>>>  # Importimi i paketes per vektorizim
 
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> vectorizer = TfidfVectorizer(min_df = 2,
                             max_df = 0.5,
                             sublinear_tf = True,
                             use_idf = True)
>>> # train_vectors ben pershtatjen dhe transformimin e datasetit me komentet
>>> train_vectors = vectorizer.fit_transform(trainData['Content'])
>>> test_vectors = vectorizer.transform(testData['Content'])
>>> # importimi i paketes Support Vector Machine dhe Confusion Matrix
>>> from sklearn import svm
>>> from sklearn.metrics import classification_report
>>> from sklearn.metrics import confusion_matrix
>>> from sklearn.metrics import accuracy_score
>>> # Krijimi i nje variable qe permban linear SVM
>>> classifier_linear = svm.SVC(kernel='linear')
>>> import time
>>> t0 = time.time()
>>> # Kemi kryer importimin e kohes
>>> # Trajnon datasetin dhe ndan se bashku me labelat perkates
>>> classifier_linear.fit(train_vectors, trainData['Label'])
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
  kernel='linear', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False)
>>> t1 = time.time()
>>> # prediction_linear mbledh parashikimin nga dataseti per testim
>>> prediction_linear = classifier_linear.predict(test_vectors)
>>> # Variabla results permban matricen e konfuzionit me labelat aktual dhe vlerat e parashikuara
>>> results = confusion_matrix(testData['Label'], prediction_linear)
>>> print(results)
[[5 3]
 [1 6]]
>>> # Shtypim saktesine ne modelin Linear SVM
>>> print('Accuracy Score: ', accuracy_score(testData['Label'], prediction_linear))
Accuracy Score:  0.7333333333333333
>>> t2 = time.time()
>>> time_linear_train = t1-t0
>>> time_linear_predict = t2-t1
>>> # Matja e kohes per trajnim dhe parashikim
>>> print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
Training time: 79.964844s; Prediction time: 131.986328s
>>> report = classification_report(testData['Label'], prediction_linear, output_dict=True)
>>> # Rezultatet per klasifikimin e komenteveme me label positive
>>> print('positive: ', report['pos'])
positive:  {'precision': 0.6666666666666666, 'recall': 0.8571428571428571, 'f1-score': 0.75, 'support': 7}
>>> # Rezultatet per klasifikimin e komenteveme me label negative
>>> print('negative: ', report['neg'])
negative:  {'precision': 0.8333333333333334, 'recall': 0.625, 'f1-score': 0.7142857142857143, 'support': 8}
>>> 
