import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn import cross_validation
from sklearn.metrics import classification_report
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def load_file_train():
    with open('train.csv') as csv_file:
        reader = csv.reader(csv_file,delimiter=",",quotechar='"')
        reader.next()
        data =[]
        target = []
        for row in reader:
            # skip missing data
            if row[1] and row[4]:
                data.append(row[1])
                target.append(row[4])

        return data,target
        

def load_file_test():
    with open('test.csv') as csv_file:
        reader = csv.reader(csv_file,delimiter=",",quotechar='"')
        reader.next()
        data =[]
        target = []
        for row in reader:
            # skip missing data
            if row[1] and row[0]:
                data.append(row[1])
                target.append(row[0])

        return data,target

# preprocess creates the term frequency matrix for the review data set
def preprocess_unigram_bigram():
    data,target = load_file_train()
    count_vectorizer = CountVectorizer(ngram_range=(1, 2),binary='False',max_df = 0.5,max_features = 18000)
    data = count_vectorizer.fit_transform(data)
    #print np.shape(data)
    tfidf_data = TfidfTransformer(use_idf=True).fit_transform(data)
    data_test,target_test = load_file_train()
    #count_vectorizer = CountVectorizer(ngram_range=(1, 2),binary='False',max_df = 0.5,max_features = 18000)
    data_test,ids = load_file_test()
    data_test = count_vectorizer.transform(data_test)
    tfidf_test = TfidfTransformer(use_idf=True).fit_transform(data_test)
    
    return tfidf_data,tfidf_test
    #return data


def learn_model(data,target):
    # preparing data for split validation. 60% training, 40% test
    data_train,data_test,target_train,target_test = cross_validation.train_test_split(data,target,test_size=0.20,random_state=24)
    print np.shape(data_train)
    print np.shape(data_test)
    classifier = BernoulliNB().fit(data_train,target_train)
    predicted = classifier.predict(data_test)
    #evaluate_model(target_test,predicted)
    #NB.append(evaluate_model(target_test,predicted)*100)


def evaluate_model(target_true,target_predicted):
    print classification_report(target_true,target_predicted)
    print "The accuracy score is {:.2%}".format(accuracy_score(target_true,target_predicted))
    
    


def learn_model_svm(data,target):
    # preparing data for split validation. 60% training, 40% test
    data_train,data_test,target_train,target_test = cross_validation.train_test_split(data,target,test_size=0.20,random_state=24)
    print "1"
    # Perform classification with SVM, kernel=linear
    classifier_linear = svm.LinearSVC()
    #classifier_linear = svm.SVC(kernel='linear')
    #classifier_linear = svm.SVC()
    #t0 = time.time()
    classifier_linear.fit(data_train,target_train)
    print "2"
    #t1 = time.time()
    predicted = classifier_linear.predict(data_test)
    print "3"
    #t2 = time.time()
    evaluate_model(target_test,predicted)
    #SVM.append(evaluate_model(target_test,predicted)*100)

def learn_model_logistic(data,target,data_test,test_ids):
    # preparing data for split validation. 80% training, 20% test
    #data_train,data_test,target_train,target_test = cross_validation.train_test_split(data,target,test_size=0.20,random_state=43)
    # Perform classification with SVM, kernel=linear
    #classifier_linear = svm.LinearSVC()
    classifier_linear = LogisticRegression()
    #classifier_linear = svm.SVC()
    #t0 = time.time()
    classifier_linear.fit(data,target)
    #t1 = time.time()
    predicted = classifier_linear.predict(data_test)
    n = len(test_ids)
    print "User_ID,Is_Response"
    for i in range(0,n):
        print test_ids[i]+","+predicted[i]
        
    #evaluate_model(target,predicted)
    #t2 = time.time()
    #LR.append(evaluate_model(target_test,predicted)*100)
# read more about model evaluation metrics here


def apply_model(tf_idf,target,tfidf_test,test_ids):
    learn_model_logistic(tf_idf,target,tfidf_test,test_ids)

def main():
    data,target = load_file_train()
    data_test,test_ids = load_file_test()
    #print data
    #print np.shape(data_test)
    #print data[0:5]
    #print target[0:4]
    #print "--------------------- Unigram + Bigram---------------------------"
    tf_idf,tfidf_test = preprocess_unigram_bigram()
    #print np.shape(tf_idf)
    #print tf_idf
    #apply_model(tf_idf,target)
    apply_model(tf_idf,target,tfidf_test,test_ids)
    
    
main()
