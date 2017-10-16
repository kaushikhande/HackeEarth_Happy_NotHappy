

import csv

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn import cross_validation
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# review.csv contains two columns
# first column is the review content (quoted)
# second column is the assigned sentiment (positive or negative)
NB = []
SVM = []
LR = []
def load_file():
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

# preprocess creates the term frequency matrix for the review data set
def preprocess_unigram():
    data,target = load_file()
    count_vectorizer = CountVectorizer(ngram_range=(1, 1),binary='False',max_df = 0.5, max_features = 18000)
    data = count_vectorizer.fit_transform(data)
    print np.shape(data)
    tfidf_data = TfidfTransformer(use_idf=True).fit_transform(data)

    return tfidf_data
    #return data

def preprocess_bigram():
    data,target = load_file()
    count_vectorizer = CountVectorizer(ngram_range=(2, 2),binary='False',max_df = 0.5, max_features = 18000)
    data = count_vectorizer.fit_transform(data)
    print np.shape(data)
    tfidf_data = TfidfTransformer(use_idf=True).fit_transform(data)

    return tfidf_data

def preprocess_trigram():
    data,target = load_file()
    count_vectorizer = CountVectorizer(ngram_range=(3, 3),binary='False',max_df = 0.5,  max_features = 18000)
    data = count_vectorizer.fit_transform(data)
    print np.shape(data)
    tfidf_data = TfidfTransformer(use_idf=True).fit_transform(data)

    return tfidf_data

# preprocess creates the term frequency matrix for the review data set
def preprocess_bigram_trigram():
    data,target = load_file()
    count_vectorizer = CountVectorizer(ngram_range=(2, 3),binary='False',max_df = 0.5, max_features = 18000)
    data = count_vectorizer.fit_transform(data)
    print np.shape(data)
    tfidf_data = TfidfTransformer(use_idf=True).fit_transform(data)

    return tfidf_data
    #return data

def preprocess_unigram_bigram():
    data,target = load_file()
    count_vectorizer = CountVectorizer(ngram_range=(1, 2),binary='False',max_df = 0.5, max_features = 18000)
    data = count_vectorizer.fit_transform(data)
    print np.shape(data)
    tfidf_data = TfidfTransformer(use_idf=True).fit_transform(data)

    return tfidf_data

def preprocess_trigram_u_b():
    data,target = load_file()
    count_vectorizer = CountVectorizer(ngram_range=(1, 3),binary='False',max_df = 0.5, max_features = 18000)
    data = count_vectorizer.fit_transform(data)
    #print np.shape(data)
    tfidf_data = TfidfTransformer(use_idf=True).fit_transform(data)

    return tfidf_data


def learn_model(data,target):
    # preparing data for split validation. 60% training, 40% test
    data_train,data_test,target_train,target_test = cross_validation.train_test_split(data,target,test_size=0.20,random_state=43)
    classifier = BernoulliNB().fit(data_train,target_train)
    predicted = classifier.predict(data_test)
    print np.shape(predicted)
    #print target_test[0:10]
    #evaluate_model(target_test,predicted)
    NB.append(evaluate_model(target_test,predicted)*100)
    return predicted


def learn_model_svm(data,target):
    # preparing data for split validation. 60% training, 40% test
    data_train,data_test,target_train,target_test = cross_validation.train_test_split(data,target,test_size=0.20,random_state=43)
    # Perform classification with SVM, kernel=linear
    classifier_linear = svm.LinearSVC()
    #classifier_linear = svm.SVC(kernel='linear')
    #classifier_linear = svm.SVC()
    #t0 = time.time()
    classifier_linear.fit(data_train,target_train)
    #t1 = time.time()
    predicted = classifier_linear.predict(data_test)
    #print np.shape(predicted)
    #print target_test[0:10]
    #t2 = time.time()
    #evaluate_model(target_test,predicted)
    SVM.append(evaluate_model(target_test,predicted)*100)
    return predicted

def learn_model_logistic(data,target):
    # preparing data for split validation. 80% training, 20% test
    data_train,data_test,target_train,target_test = cross_validation.train_test_split(data,target,test_size=0.20,random_state=43)
    # Perform classification with SVM, kernel=linear
    #classifier_linear = svm.LinearSVC()
    classifier_linear = LogisticRegression()
    #classifier_linear = svm.SVC()
    #t0 = time.time()
    classifier_linear.fit(data_train,target_train)
    #t1 = time.time()
    predicted = classifier_linear.predict(data_test)
    #print np.shape(predicted)
    #print target_test[0:10]
    #t2 = time.time()
    LR.append(evaluate_model(target_test,predicted)*100)
    return predicted
# read more about model evaluation metrics here
# http://scikit-learn.org/stable/modules/model_evaluation.html
def evaluate_model(target_true,target_predicted):
    print classification_report(target_true,target_predicted)
    print "The accuracy score is {:.2%}".format(accuracy_score(target_true,target_predicted))
    return accuracy_score(target_true,target_predicted)

def apply_model(tf_idf,target,data):
    print "Naive Bayes"
    nb = learn_model(tf_idf,target)
    print "Support Vector Machine"
    svm = learn_model_svm(tf_idf,target)
    print "Logistic Regression"
    lr = learn_model_logistic(tf_idf,target)
    data_train,data_test,target_train,target_test = cross_validation.train_test_split(data,target,test_size=0.20,random_state=43)
    final_pred = []
    for i in range(0,7787):
        c1 = 0
        if nb[i] == 'happy':
            c1 = c1 + 1
        if lr[i] == 'happy':
            c1 = c1 + 1
        if svm[i] == 'happy':
            c1 = c1 + 1
        #print i
        if c1 == 3 or c1 == 2:
            final_pred.append('happy')
        else:
            final_pred.append('not happy')
        
    print "-----------------------"
    #print final_pred
    print "-----------------------"
    #print new_label
    
    print "Results of ensemble: NB + SVM + ME::"
    print "----------Confusion Matrix--------------"
    print classification_report(target_test,final_pred)
    print ""
    print "The accuracy score of ensemble is {:.2%}".format(accuracy_score(target_test,final_pred))
    print "##############################################"
    

    
    #print nb[0:10],svm[0:10],lr[0:10]
    
def graph():
    N = 3
    ind = np.arange(N)  # the x locations for the groups
    width = 0.20       # the width of the bars
    
    fig, ax = plt.subplots()
    NB1 = (NB[0],NB[1],NB[2])
    
    SVM1 = (SVM[0],SVM[1],SVM[2])
    LR1 = (LR[0],LR[1],LR[2])
    rects1 = ax.bar(ind, NB1, width, color='r')

    rects2 = ax.bar(ind + width, SVM1, width, color='y')

    rects3 = ax.bar(ind + width*2, LR1, width, color='g')

    Ensembles_values = (82.28, 82.69, 82.48)
    print Ensembles_values
    
    rects4 = ax.bar(ind + width*3, Ensembles_values, width, color='b')


    # add some text for labels, title and axes ticks
    ax.set_ylabel('Scores')
    ax.set_title('Scores by Features and accuracy')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(('Unigram', 'Bigram', 'trigram'))
    plt.ylim([75,88])

    ax.legend((rects1[0], rects2[0], rects3[0],rects4[0]), ('Naive Bayes', 'SVM', 'ME', 'Ensemble'))    


    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.00*height,
                    '%.2f' % (height),
                    ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    plt.show()


    
def main():
    data,target = load_file()
    
    print "--------------------- Unigram ---------------------------"
    tf_idf = preprocess_unigram()
    apply_model(tf_idf,target,data)
    print "--------------------- Bigram ---------------------------"
    tf_idf = preprocess_bigram()
    apply_model(tf_idf,target,data)
    print "----------------------- Trigram -------------------------"
    tf_idf = preprocess_trigram()
    apply_model(tf_idf,target,data)
    print "--------------------- Unigram + Bigram---------------------------"
    tf_idf = preprocess_unigram_bigram()
    apply_model(tf_idf,target,data)
    print "--------------------- Bigram + Trigram---------------------------"
    tf_idf = preprocess_bigram_trigram()
    apply_model(tf_idf,target,data)
    print "----------------------- Trigram+Unigram+Bigram -------------------------"
    tf_idf = preprocess_trigram_u_b()
    apply_model(tf_idf,target,data)

    print NB
    print SVM
    print LR
    graph()


main()

