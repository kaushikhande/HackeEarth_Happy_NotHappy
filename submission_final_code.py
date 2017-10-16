import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import cross_validation
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


def load_file_train():
    with open('train.csv') as csv_file:
        reader = csv.reader(csv_file,delimiter=",",quotechar='"')
        reader.next()
        data =[]
        target = []
        for row in reader:
            # Reads description and Is_Response from train.csv
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
            # Reads description and User_ID from test.csv
            if row[1] and row[0]:
                data.append(row[1])
                target.append(row[0])

        return data,target

# Creates unigram + bigram document term matrix
def preprocess_unigram_bigram():
    data,target = load_file_train()
    count_vectorizer = CountVectorizer(ngram_range=(1, 2),binary='False',max_df = 0.5,max_features = 18000)
    data = count_vectorizer.fit_transform(data)
    tfidf_data = TfidfTransformer(use_idf=True).fit_transform(data)
    data_test,target_test = load_file_train()
    data_test,ids = load_file_test()
    data_test = count_vectorizer.transform(data_test)
    tfidf_test = TfidfTransformer(use_idf=True).fit_transform(data_test)
    
    return tfidf_data,tfidf_test


def evaluate_model(target_true,target_predicted):
    print classification_report(target_true,target_predicted)
    print "The accuracy score is {:.2%}".format(accuracy_score(target_true,target_predicted))
    
    


def learn_model_logistic(data,target,data_test,test_ids):
    #-----LogisticRegression-----
    classifier = LogisticRegression()
    classifier.fit(data,target)
    predicted = classifier.predict(data_test)
    n = len(test_ids)
    print "User_ID, Is_Response"
    for i in range(0,n):
        print test_ids[i]+","+predicted[i]
        

def apply_model(tf_idf,target,tfidf_test,test_ids):
    learn_model_logistic(tf_idf,target,tfidf_test,test_ids)

def main():
    data,target = load_file_train()
    data_test,test_ids = load_file_test()
    #--------------------- Unigram + Bigram---------------------------"
    tf_idf,tfidf_test = preprocess_unigram_bigram()
    apply_model(tf_idf,target,tfidf_test,test_ids)
    
    
main()
