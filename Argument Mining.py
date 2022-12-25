import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import time
from sklearn.model_selection import GridSearchCV

# following function was used to tunning the parameters of SVM machine 
def finding_best_params(X_train,y_train,X_test,y_test):
    param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid','linear']}
    grid = GridSearchCV(svm.SVC(),param_grid,refit=True,verbose=2,cv=2)
    grid.fit(X_train,y_train)
    print(grid.best_estimator_)
    grid_predictions = grid.predict(X_test)
    print(confusion_matrix(y_test,grid_predictions))
    print(classification_report(y_test,grid_predictions))

start_time = time.time()
stop_words = set(stopwords.words('english'))
word_lemmatizer = WordNetLemmatizer()
pathTest =  "./data/test-bio.csv"
pathTrain =  "./data/train-bio.csv"

CorpusTest = pd.read_csv(pathTest , sep='\t', header=None)
CorpusTrain = pd.read_csv(pathTrain, sep='\t', header=None)

# Data Pre-processing  
Saperators = ['_END_PARAGRAPH' , 'END_ESSAY_','?', ',','.','!',';',':']

CorpusTest = CorpusTest[~CorpusTest[0].isin(Saperators) ]
CorpusTrain = CorpusTrain[~CorpusTrain[0].isin(Saperators) ]

#remove blank space
CorpusTest[0].dropna(inplace=True)
CorpusTrain[0].dropna(inplace=True)

#converting float to string
CorpusTest[0] = [str(entry) for entry in CorpusTest[0]]
CorpusTrain[0] = [str(entry) for entry in CorpusTrain[0]]

#Lowercase the text
CorpusTest[0] = [entry.lower() for entry in CorpusTest[0]]
CorpusTrain[0] = [entry.lower() for entry in CorpusTrain[0]]

#remove stopwords
CorpusTest = CorpusTest[~CorpusTest[0].isin(stop_words)]
CorpusTrain = CorpusTrain[~CorpusTrain[0].isin(stop_words)]

#lable lemmatization
CorpusTest[0] = [word_lemmatizer.lemmatize(entry) for entry in CorpusTest[0]]
CorpusTrain[0] = [word_lemmatizer.lemmatize(entry) for entry in CorpusTrain[0]]

Test_csv = pd.DataFrame(CorpusTest)
Test_csv.to_csv('./data/test-bio.csv',header=False,index=False,sep='\t')

Train_X = CorpusTrain[0]
Train_Y = CorpusTrain[1]

Test_X = CorpusTest[0].tolist()
Test_Y = CorpusTest[1].tolist()

Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

#Vectorization of corpus using TF-IDF, Features extraction
Tfidf_vec = TfidfVectorizer()
Tfidf_vec.fit(Train_X)

Train_X_Tfidf = Tfidf_vec.transform(Train_X)
Test_X_Tfidf = Tfidf_vec.transform(Test_X)


#parameter tunning in the following function

#finding_best_params(Train_X_Tfidf,Train_Y,Test_X_Tfidf,Test_Y)


# Naive bayes model implementation with tf-idf

    #Naive = naive_bayes.MultinomialNB()
    #Naive.fit(Train_X_Tfidf,Train_Y)
    ## predict the labels on validation dataset
    #predictions = Naive.predict(Test_X_Tfidf)


# SVM model implementation with tf-IDF

SVM = svm.SVC(C=1, gamma=1, kernel='sigmoid')
SVM.fit(Train_X_Tfidf, Train_Y)
predictions = SVM.predict(Test_X_Tfidf)

#print(classification_report(Test_Y,predictions,labels=np.unique(predictions)))

predictions_decoded= Encoder.inverse_transform(predictions)

#print("y_pred_decoded: ",predictions_decoded)
    
predict_csv = {
        'token': [],
        'label': []
    }
    
for i in range(len(predictions_decoded)):
        predict_csv['token'].append(str(Test_X[i]))
        predict_csv['label'].append(str(predictions_decoded[i]))
    
extract_file= pd.DataFrame(predict_csv)
extract_file.to_csv('predictions.csv',header=False,index=False,sep='\t')

#timer to check the execution time
print("Process finished --- %s seconds ---" % (time.time() - start_time))