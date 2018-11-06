import os 
import re
import nltk
from nltk import PorterStemmer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy.random as nr
import sklearn.metrics as sklm


# read the body of each email and store it:
spam = []
ham = []

for filename in os.listdir('spam'):
    spamf = open(os.path.join('spam', filename)).read()
    spam.append(spamf[spamf.find('\n\n'):])

for filename in os.listdir('ham'):
    hamf = open(os.path.join('ham', filename)).read()
    ham.append(hamf[hamf.find('\n\n'):])

# clean emails :
ps = PorterStemmer()
stop_words = nltk.corpus.stopwords.words('english')


def normalization(text):
    '''
    function to clean and normalize emails.
    '''
    # stripping html tags:
    text = re.sub('<[^<>]+>', ' ', text)
    # normalize numbers:
    text = re.sub('\d+', 'number', text) 
    # normalize emails:
    text = re.sub('[^\s]+@[^\s]+', 'emailaddr', text)
    # normalize urls:
    text = re.sub('(http|https)://[^\s]*', 'httpaddr', text)
    # normalize dollar sign:
    text = re.sub('$', 'dollar', text)
    # remove non-words and lower casing:
    text = re.sub('[_]+', ' ', text)
    text = re.findall('\w+', text.lower())
    # remove stop words:
    text = [word for word in text if word not in stop_words]
    # word stemming
    text = [ps.stem(word) for word in text]
    return text


spam = list(map(normalization, spam))
ham = list(map(normalization, ham))

# choose most frequent words and creat vocabulary dictionary:
all_mails = spam + ham 
all_words = []

for mail in all_mails:
    for word in mail:
        all_words.append(word)

uniques = set(all_words)
count_words = {}

for word in uniques:
    count_words[word] = all_words.count(word)

most_freq = sorted(count_words, key=count_words.get, reverse=True)[:1500]
vocab_list = dict(zip(range(len(most_freq)), most_freq))


def word_indices(text):
    '''
    function that returns a list of indices of the words
    contained in the email.
    '''
    word_indices = []
    for word in text:
        for i in vocab_list:
            if word == vocab_list[i]:
                word_indices.append(i)
                break
    return word_indices


spam_ind = list(map(word_indices, spam))
ham_ind = list(map(word_indices, ham))


# creat features and label for Emails: 


def features(word_ind):
    '''
    function that takes a word_indices vector and produces a feature vector
    from the word indices.
    '''
    features = []
    for idx in vocab_list:
        if idx in word_ind:
            features.append(1)
        else:
            features.append(0)
    return features


spam_features = np.array(list(map(features, spam_ind)))
ham_features = np.array(list(map(features, ham_ind)))
spam_label = np.array([1 for _ in range(spam_features.shape[0])])
ham_label = np.array([0 for _ in range(ham_features.shape[0])])
Features = np.concatenate((spam_features, ham_features), axis=0)
Labels = np.concatenate((spam_label, ham_label), axis=0)

# split data:
nr.seed(1115)
indx = range(Features.shape[0])
indx = train_test_split(indx, test_size=0.2)
X_train = Features[indx[0]]
y_train = np.ravel(Labels[indx[0]])
X_test = Features[indx[1]]
y_test = np.ravel(Labels[indx[1]])

# define and fit a linear SVM model:
nr.seed(1115)
svm_mod = svm.LinearSVC(C=0.1)
svm_mod.fit(X_train, y_train)

# evaluate the model results:


def print_metrics(labels, scores):
    '''
    function that takes label, scores and prints (confusion-matrix, accuracy,
                                                  precition, recall, f-score).
    ''' 
    metrics = sklm.precision_recall_fscore_support(labels, scores)
    conf = sklm.confusion_matrix(labels, scores)
    print('                 Confusion matrix')
    print('                 Score positive    Score negative')
    print('Actual positive    %6d' % conf[1,1] + '             %5d' % conf[1,0])
    print('Actual negative    %6d' % conf[0,1] + '             %5d' % conf[0,0])
    print('')
    print('Accuracy        %0.2f' % sklm.accuracy_score(labels, scores))
    print('Macro precision %0.2f' % float((float(metrics[0][0]) + float(metrics[0][1]))/2.0))
    print('Macro recall    %0.2f' % float((float(metrics[1][0]) + float(metrics[1][1]))/2.0))
    print(' ')
    print('           Negative      Positive')
    print('Num case   %6d' % metrics[3][0] + '        %6d' % metrics[3][1])
    print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1])
    print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1])
    print('F1         %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1])
    

score = svm_mod.predict(X_test)
print_metrics(y_test, score)    