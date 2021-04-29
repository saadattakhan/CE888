import pandas as pd
import numpy as np
from utils import Utils
import collections

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import feature_extraction

import spacy
import en_core_web_sm
import string

import warnings
warnings.filterwarnings("ignore")

class Data:
    #### Constructor. Setting dataset folder path, labels and their values
    #### Loading dataset in pandas dataframe for train, validation and test sets
    def __init__(self,path,task):
        if(task == "hate"):
            labels=[0,1]
            labels_info=["Not-Hate","Hate"]
        elif(task=="offensive"):
            labels=[0,1]
            labels_info=["Not-Offensive","Offensive"]
        elif(task=="sentiment"):
            labels=[0,1,2]
            labels_info=["Negative","Neutral","Positive"]
        else:
            raise NotImplementedError("Method not implemented")

        self.task=task
        self.labels=labels
        self.labels_info=labels_info

        try:
            with open(path+"train_text.txt","r", encoding="utf8") as f:
                train_text=f.read()
            with open(path+"train_labels.txt","r", encoding="utf8") as f:
                train_labels=f.read()
            with open(path+"test_text.txt","r", encoding="utf8") as f:
                test_text=f.read()
            with open(path+"test_labels.txt","r", encoding="utf8") as f:
                test_labels=f.read()
            with open(path+"val_text.txt","r", encoding="utf8") as f:
                val_text=f.read()
            with open(path+"val_labels.txt","r", encoding="utf8") as f:
                val_labels=f.read()
            train_text=train_text.split("\n")
            train_labels=train_labels.split("\n")

            train_text=train_text[:-1]
            train_labels=train_labels[:-1]


            data={"text":train_text,"label":train_labels}

            train=pd.DataFrame(data,columns=["text","label"])


            test_text=test_text.split("\n")
            test_labels=test_labels.split("\n")

            test_text=test_text[:-1]
            test_labels=test_labels[:-1]


            data={"text":test_text,"label":test_labels}

            test=pd.DataFrame(data,columns=["text","label"])


            val_text=val_text.split("\n")
            val_labels=val_labels.split("\n")

            val_text=val_text[:-1]
            val_labels=val_labels[:-1]


            data={"text":val_text,"label":val_labels}

            validation=pd.DataFrame(data,columns=["text","label"])
        except Exception as e:
            raise FileNotFoundError("Cannot find dataset files. Check path")
        self.train=train
        self.validation=validation
        self.test=test


    #### Transforming words dataset to vector form to feed to machine learning models
    #### Chossing best models (after research) for classification tasks
    #### Bag of Words, TF-IDF and Spacy vectors
    #### Feature Selection for BOW and TFIDF using Chi Statistic
    def split(self,selection):
        print("Extracting features from data...")
        if(self.task=="hate"):
            print("Using spacy vectors")
            feature="spacy"
            self._best_model="rf"
        elif(self.task=="offensive"):
            print("Using TF-IDF vectors")
            feature="tfidf"
            self._best_model="rf"
        elif(self.task=="sentiment"):
            print("Using spacy vectors")
            feature="spacy"
            self._best_model="lr"
        else:
            print("Using TF-IDF vectors")
            feature="tfidf"
            self._best_model="lr"

        if(feature=="bow"):
            vectorizer = CountVectorizer()


            X_train = vectorizer.fit_transform(self.train.text.values).toarray()
            self.train.label=pd.to_numeric(self.train.label)
            y_train = self.train.loc[:,'label'].values

            if(selection):
                X_words=vectorizer.get_feature_names()
                features = pd.DataFrame()
                for label in np.unique(self.y_train):
                    chi2, p = feature_selection.chi2(self.X_train, self.y_train==label)
                    features = features.append(pd.DataFrame(
                                   {"words":X_words, "score":1-p, "y":label}))
                    features = features.sort_values(["y","score"],ascending=[True,False])
                    features = features[features["score"]>0.95]
                X_words = features["words"].unique().tolist()

                vectorizer = CountVectorizer(vocabulary=X_words)
                X_train = vectorizer.fit_transform(self.train.text.values).toarray()


            X_val = vectorizer.transform(self.validation.text.values).toarray()
            self.validation.label=pd.to_numeric(self.validation.label)
            y_val = self.validation.loc[:,'label'].values

            X_test = vectorizer.transform(self.test.text.values).toarray()
            self.test.label=pd.to_numeric(self.test.label)
            y_test = self.test.loc[:,'label'].values
        elif(feature=="tfidf"):
            vectorizer = TfidfVectorizer()

            X_train = vectorizer.fit_transform(self.train.text.values).toarray()
            self.train.label=pd.to_numeric(self.train.label)
            y_train = self.train.loc[:,'label'].values

            if(selection):
                X_words=vectorizer.get_feature_names()
                features = pd.DataFrame()
                for label in np.unique(self.y_train):
                    chi2, p = feature_selection.chi2(self.X_train, self.y_train==label)
                    features = features.append(pd.DataFrame(
                                   {"words":X_words, "score":1-p, "y":label}))
                    features = features.sort_values(["y","score"],ascending=[True,False])
                    features = features[features["score"]>0.95]
                X_words = features["words"].unique().tolist()

                vectorizer = TfidfVectorizer(vocabulary=X_words)
                X_train = vectorizer.fit_transform(self.train.text.values).toarray()

            X_val = vectorizer.transform(self.validation.text.values).toarray()
            self.validation.label=pd.to_numeric(self.validation.label)
            y_val = self.validation.loc[:,'label'].values

            X_test = vectorizer.transform(self.test.text.values).toarray()
            self.test.label=pd.to_numeric(self.test.label)
            y_test = self.test.loc[:,'label'].values
        elif(feature=="spacy"):
            nlp = spacy.load('en_core_web_md')
            X_train=self.train.text.apply(lambda x:nlp(x).vector).values
            X_train=np.stack(X_train)
            self.train.label=pd.to_numeric(self.train.label)
            y_train = self.train.loc[:,'label'].values


            X_val=self.validation.text.apply(lambda x:nlp(x).vector).values
            X_val=np.stack(X_val)
            self.validation.label=pd.to_numeric(self.validation.label)
            y_val = self.validation.loc[:,'label'].values

            X_test=self.test.text.apply(lambda x:nlp(x).vector).values
            X_test=np.stack(X_test)
            self.test.label=pd.to_numeric(self.test.label)
            y_test = self.test.loc[:,'label'].values
        else:
            raise NotImplementedError("Method not implemented")
        return X_train,y_train,X_val,y_val,X_test,y_test


    #### Pre-Processing dataset
    #### Remove unnecessary words and characters from the tweets

    def preprocess(self,selection=False):

        print("Preprocessing dataset...")

        nlp=en_core_web_sm.load()

        self.train.text=self.train.text.apply(lambda x: Utils.expand_contractions(x))
        self.train.text=self.train.text.apply(lambda x: str(x).lower())
        self.train.text=self.train.text.apply(lambda x: Utils.convert_emoticons(x))
        self.train.text=self.train.text.apply(lambda x: Utils.convert_emojis(x))
        self.train.text=self.train.text.apply(lambda x: Utils.hashtag_handling(x))
        self.train.text=self.train.text.apply(lambda x: Utils.remove_handles(x))
        self.train.text=self.train.text.apply(lambda x: Utils.clean_text(x))
        self.train.text=self.train.text.apply(lambda x: Utils.nlp_process(nlp,x))
        self.train.text=self.train.text.apply(lambda x: Utils.trim_text(x))
        self.train.text=self.train.text.apply(lambda x: Utils.remove_short_words(x))

        self.validation.text=self.validation.text.apply(lambda x: Utils.expand_contractions(x))
        self.validation.text=self.validation.text.apply(lambda x: str(x).lower())
        self.validation.text=self.validation.text.apply(lambda x: Utils.convert_emoticons(x))
        self.validation.text=self.validation.text.apply(lambda x: Utils.convert_emojis(x))
        self.validation.text=self.validation.text.apply(lambda x: Utils.hashtag_handling(x))
        self.validation.text=self.validation.text.apply(lambda x: Utils.remove_handles(x))
        self.validation.text=self.validation.text.apply(lambda x: Utils.clean_text(x))
        self.validation.text=self.validation.text.apply(lambda x: Utils.nlp_process(nlp,x))
        self.validation.text=self.validation.text.apply(lambda x: Utils.trim_text(x))
        self.validation.text=self.validation.text.apply(lambda x: Utils.remove_short_words(x))

        self.test.text=self.test.text.apply(lambda x: Utils.expand_contractions(x))
        self.test.text=self.test.text.apply(lambda x: str(x).lower())
        self.test.text=self.test.text.apply(lambda x: Utils.convert_emoticons(x))
        self.test.text=self.test.text.apply(lambda x: Utils.convert_emojis(x))
        self.test.text=self.test.text.apply(lambda x: Utils.hashtag_handling(x))
        self.test.text=self.test.text.apply(lambda x: Utils.remove_handles(x))
        self.test.text=self.test.text.apply(lambda x: Utils.clean_text(x))
        self.test.text=self.test.text.apply(lambda x: Utils.nlp_process(nlp,x))
        self.test.text=self.test.text.apply(lambda x: Utils.trim_text(x))
        self.test.text=self.test.text.apply(lambda x: Utils.remove_short_words(x))


        print("Dataset preprocessing completed successfully")

        return self.split(selection)

    #### Get all the words in training set after pre-processing
    def get_words(self,label):
        return ' '.join([text for text in self.train['text'][self.train['label'] == label]])
