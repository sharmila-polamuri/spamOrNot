"""
Function to find spam or not spam of text using nltk
"""
import sys
import nltk
import sklearn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn import model_selection
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print("Vesrions of imported libraries")
print(f"Python  :- {sys.version}")
print(f"NLTK :- {nltk.__version__}")
# print(f"Scikit-learn :- {sklearn.vesrion()}")
print(f"Pandas :- {pd.__version__}")
print(f"Numpy :- {np.__version__}")

def load_data(datapath):
    """
    Input :- path of file
    output :-  Load file to pandas dataframe
    """
    dataset = pd.read_table(datapath, header=None, encoding="utf-8")
    return dataset

def preprocessing(labels, text):
    """
    Input :- lables and text field
    Output :- encoded labels and preprocessing text
    """
    encoder = LabelEncoder()
    Y = encoder.fit_transform(labels)
    # replace email links with email 
    processed = text.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$','emailaddress')
    # Replace URLs with 'webaddress'
    processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$',
                                  'webaddress')
    # Replace money symbols with 'moneysymb' (£ can by typed with ALT key + 156)
    processed = processed.str.replace(r'£|\$', 'moneysymb')
    # Replace 10 digit phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'phonenumber'
    processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$',
                                  'phonenumbr')
    # Replace numbers with 'numbr'
    processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')
    # Remove punctuation
    processed = processed.str.replace(r'[^\w\d\s]', ' ')
    # Replace whitespace between terms with a single space
    processed = processed.str.replace(r'\s+', ' ')
    # Remove leading and trailing whitespace
    processed = processed.str.replace(r'^\s+|\s+?$', '')
    # change words to lower case - Hello, HELLO, hello are all the same word
    processed = processed.str.lower()

    return Y, processed

def raw_text_preprocessing(text):
    """
    Input :- lables and text field
    Output :- encoded labels and preprocessing text
    """
    # replace email links with email 
    processed = text.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$','emailaddress')
    # Replace URLs with 'webaddress'
    processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$',
                                  'webaddress')
    # Replace money symbols with 'moneysymb' (£ can by typed with ALT key + 156)
    processed = processed.str.replace(r'£|\$', 'moneysymb')
    # Replace 10 digit phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'phonenumber'
    processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$',
                                  'phonenumbr')
    # Replace numbers with 'numbr'
    processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')
    # Remove punctuation
    processed = processed.str.replace(r'[^\w\d\s]', ' ')
    # Replace whitespace between terms with a single space
    processed = processed.str.replace(r'\s+', ' ')
    # Remove leading and trailing whitespace
    processed = processed.str.replace(r'^\s+|\s+?$', '')
    # change words to lower case - Hello, HELLO, hello are all the same word
    processed = processed.str.lower()

    return processed

def remove_stopwords(preprocess_data):
    """
    remove stop words from input text
    """
    stop_words = set(stopwords.words('english'))
    preprocess_data = preprocess_data.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))
    return preprocess_data
def word_stem(preprocess_data):
    """
    applying word stemming technique to text
    """
    porterstem = nltk.PorterStemmer()
    preprocess_data = preprocess_data.apply(lambda x:' '.join(porterstem.stem(term) for term in x.split()))
    return preprocess_data
def collect_tokens(preprocess_data):
    """
    Input :- preprocess data 
    Output :-  word tokens
    """
    all_words = []
    for sent in preprocess_data:
        words = word_tokenize(sent)
        for word in words:
            all_words.append(word)
    all_words = nltk.FreqDist(all_words)
    return all_words
def find_features(sent,word_features):
    """
    The find_features function will determine which of the 1500 word features are contained in the review
    """
    words = word_tokenize(sent)
    features = {}
    for word in word_features:
        features[word] = (word in words)
    return features

def svc_model(training, testing):
    """
    Training a svc model from nltk
    """
    # model = SklearnClassifier(SVC(kernal = 'linear'))
    model = SklearnClassifier(SVC(kernel = 'linear'))
    # train the model on training data
    model.train(training)
    # test model on testing data
    accuracy = nltk.classify.accuracy(model, testing)*100
    # result = model.classify("You won 100000000 rupee")
    return accuracy
def diff_models(training, testing):
    """
    train different models on training data and then test on testing data
    """
    # Define models to train
    names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes", "SVM Linear"]
    
    classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
    ]
    models = list(zip(names, classifiers))
    for name, model in models:
        nltk_model = SklearnClassifier(model)
        nltk_model.train(training)
        accuracy = nltk.classify.accuracy(nltk_model, testing)*100
        print("{} Accuracy: {}".format(name, accuracy))


if __name__=="__main__":
    """
    Main function of spam filter model
    """
    dataset = load_data("smsspamcollection/SMSSpamCollection")
    print(dataset.head())
    print(dataset.info())
    # count of sentences of each label
    labels = dataset[0]
    print(labels.value_counts())
    # data preprocessing
    Y, preprocess_data = preprocessing(labels, dataset[1])
    print(Y[:5])
    print(preprocess_data[:5])
    # remove stop words
    preprocess_data = remove_stopwords(preprocess_data)
    print(preprocess_data[:5])
    # apply stemming
    preprocess_data = word_stem(preprocess_data)
    print(preprocess_data[:5])
    # GATHERING features
    all_words = collect_tokens(preprocess_data)
    print(f"Number of words {len(all_words)}")
    print(f"Most common words {all_words.most_common(15)}")
    # user top most 1500 words as features
    word_features = list(all_words.keys())[:1500]
    # find features
    all_data_zip = list(zip(preprocess_data,Y))
    print(f"after all data zip {preprocess_data[0]}")
    # define a seed for reproducibility
    seed = 1
    np.random.seed = seed
    np.random.shuffle(all_data_zip)
    #call find_features function for each SMS message
    featuresets = [(find_features(text, word_features), label)for (text, label) in all_data_zip]
    # split data into train and test data
    training, testing = model_selection.train_test_split(featuresets, test_size = 0.25, random_state=seed)
    print(len(training))
    print(len(testing))
    # models training starts here
    svc_accuracy = svc_model(training, testing)
    print(f"Accuracy :- {svc_accuracy}")
    # train on different models
    diff_models(training, testing)

    # predict new raw text data
    prediting_text = pd.Series(["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"])
    raw_text_preprocess = raw_text_preprocessing(prediting_text)
    raw_text_preprocess = remove_stopwords(raw_text_preprocess)
    raw_text_preprocess = word_stem(raw_text_preprocess)
    raw_text_preprocess = collect_tokens(raw_text_preprocess)
    raw_text = ""
    for text in raw_text_preprocess:
        raw_text = raw_text + text + " "
    raw_text = raw_text.strip()
    print(raw_text)
    # print(f"raw text preprocess {raw_text_preprocess[0]}")
    # all_raw_data = list(zip(raw_text_preprocess))
    # raw_features = [find_features(text, word_features)for text in all_raw_data]
    features = find_features(raw_text,word_features)
    model = SklearnClassifier(SVC(kernel = 'linear'))
    classifier = model.train(training)
    predict_result = classifier.classify(features)
    print(f"predict result {predict_result}")
