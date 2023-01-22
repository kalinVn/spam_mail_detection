import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer

import config
from factory.Classifier import Classifier
from service.PreprocessText import PreprocessText

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class SpamMailDetection:

    def __init__(self):

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.logistic_model_bow = None
        self.logistic_model_tfidf = None

        self.x_train_vectorized = None
        self.x_test_vectorized = None

        csv_path = config.CSV_PATH
        self.dataset = pd.read_csv(csv_path)
        self.preprocess_text = PreprocessText(self.dataset)

        self.classifier_factory = Classifier()
        self.model = self.classifier_factory.get_model()
        self.count_vectorized = CountVectorizer(min_df=0, max_df=1, binary=False, ngram_range=(1, 3))
        self.vectorized = self.classifier_factory.get_vectorized()

        self.label_b = LabelBinarizer()

    def get_dataset(self):
        return self.dataset

    def preprocess(self):
        self.dataset.loc[self.dataset['Category'] == 'spam', 'Category',] = 0
        self.dataset.loc[self.dataset['Category'] == 'ham', 'Category',] = 1

        self.dataset['Message'] = self.dataset['Message'].apply(self.preprocess_text.noise_remove_vals)
        self.dataset['Message'] = self.dataset['Message'].apply(self.preprocess_text.stemmer)
        self.dataset['Message'] = self.dataset['Message'].apply(self.preprocess_text.removing_stopwords)

        self.x_train = self.dataset.Message[:config.TRAINING_DATA_SIZE]
        self.x_test = self.dataset.Message[config.TRAINING_DATA_SIZE:]

        self.x_train_vectorized = self.vectorized.fit_transform(self.x_train)
        self.x_test_vectorized = self.vectorized.transform(self.x_test)


        # print(dataset.head())

        # separating the data as texts and label
        # print(self.dataset)

        self.y_train = self.dataset.Category[:config.TRAINING_DATA_SIZE]
        self.y_test = self.dataset.Category[config.TRAINING_DATA_SIZE:]

        self.y_train = self.y_train.astype('int')
        self.y_test = self.y_test.astype('int')

    def build(self):
        self.model.fit(self.x_train_vectorized, self.y_train)

    def test_accuracy_score(self):
        prediction_x_train = self.model.predict(self.x_train_vectorized)

        result_accuracy_train_data = accuracy_score(self.y_train, prediction_x_train)
        print("Accuracy score of tfidf training data: ", result_accuracy_train_data)

        prediction_x_test = self.model.predict(self.x_test_vectorized)
        accuracy_on_test_data = accuracy_score(self.y_test, prediction_x_test)
        print("Accuracy on test data: ", accuracy_on_test_data)

    def predict(self, mail_message):
        dataset = pd.DataFrame({'Message': mail_message})

        dataset['Message'] = dataset['Message'].apply(self.preprocess_text.noise_remove_vals)
        dataset['Message'] = dataset['Message'].apply(self.preprocess_text.stemmer)
        dataset['Message'] = dataset['Message'].apply(self.preprocess_text.removing_stopwords)
        input_data_features = self.vectorized.transform(dataset)

        prediction = self.model.predict(input_data_features)

        if prediction[0] == 1:
            print("The email is ham")
        else:
            print("The email is spam")


