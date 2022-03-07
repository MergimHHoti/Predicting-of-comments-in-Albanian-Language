Python 3.7.3 (v3.7.3:ef4ec6ed12, Mar 25 2019, 21:26:53) [MSC v.1916 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import pandas as pd
>>> trainData = pd.read_csv ('D:\Mërgim H. HOTI-PhD Studies at SEEU\First year 2018-2019\Second Semester 2018-2019\Advanced Topics in Information and Communication Technologies - Arbana Kadriu\Project\Implementation part\KomentetTrainF.csv') # # trainData merr datasetin
>>> testData = pd.read_csv ('D:\Mërgim H. HOTI-PhD Studies at SEEU\First year 2018-2019\Second Semester 2018-2019\Advanced Topics in Information and Communication Technologies - Arbana Kadriu\Project\Implementation part\KomentetTest.csv') # TestData merr datasetin per Testim
>>> from sklearn.feature_extraction.text import TfidfVectorizer # Importimi i paketes per vektorizim
>>> vectorizer = TfidfVectorizer(min_df = 2,
			     max_df = 0.5,
			     sublinear_tf = True,
			     use_idf = True)
>>> train_vectors = vectorizer.fit_transform(trainData['Content']) # train_vectors bene pershtatjen dhe transformimin e datasetit me komentet
>>> test_vectors = vectorizer.transform(testData['Content']) # test_vectors bene testimin e datasetit me komentet ekzistuese
>>> # Hapat e meposhtem bejne importimin i paketes Support Vector Machine dhe Confusion Matrices
>>> from sklearn import svm
>>> from sklearn.metrics import classification_report
>>> from sklearn.metrics import confusion_matrix
>>> from sklearn.metrics import accuracy_score
>>> classifier_linear = svm.SVC(kernel='linear') # Krijimi i nje variable qe permban SVM-n lineare
>>> import time
>>> t0 = time.time() # Rreshti me larte mundeson importimin per matjen e kohes se ekzekutimit te parashikimit, ndersa rreshti i tanishem mundeson paraqitjen e kohes pas ekzekutimit te klasifikimit te saj
>>> classifier_linear.fit(train_vectors, trainData['Label'])
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
  kernel='linear', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False)
>>> t1 = time.time()
>>> prediction_linear = classifier_linear.predict(test_vectors) # prediction_linear mbledh parashikimin nga dataseti per Testim
>>> results = confusion_matrix(testData['Label'], prediction_linear) # Variabla results permban Matricen e Konfuzionit me labelat aktuale dhe vlerat e parashikuara
>>> print(results) # Variabla print mundeson paraqitjen e te rezultateve ne Matric 2 me 2
[[18 20]
 [ 9 30]]
>>> print('Accuracy Score: ', accuracy_score(testData['Label'], prediction_linear)) # Shtypim saktesine ne modelin SVM linear
Accuracy Score:  0.6233766233766234
>>> # Hapat e me poshtem mundesojne Matjen e Kohes per trajnim dhe parashikim te komenteve
>>> t2 = time.time()
>>> time_linear_train = t1-t0
>>> time_linear_predict = t2-t1
>>> print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict)) #Paraqitja e rezultatit per Kohen, Precizitetin, Parashikimin linear kohor
Training time: 16.842594s; Prediction time: 52.563069s
>>> report = classification_report(testData['Label'], prediction_linear, output_dict=True) # Paraqitja e raportit klasifikues
>>> print('positive: ', report['pos'])
positive:  {'precision': 0.6, 'recall': 0.7692307692307693, 'f1-score': 0.6741573033707865, 'support': 39}
>>> # Ne rreshtin me larte jane te paraqitura rezultatet per komentet pozitive
>>> print('negative: ', report['neg']) # Ne kete pjese jane te paraqitura komentet negative
negative:  {'precision': 0.6666666666666666, 'recall': 0.47368421052631576, 'f1-score': 0.5538461538461538, 'support': 38}
>>> 
