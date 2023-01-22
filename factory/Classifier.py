from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import config


class Classifier:

    def __init__(self):
        self.name = config.MODEL

    def get_model(self):
        if self.name == "svm":
            return svm.SVC()
        elif self.name == "logistic_regression":
            return LogisticRegression(penalty='l2', max_iter=500, C=1, random_state=42)
        elif self.name == "linear_regression":
            return LinearRegression()
        else:
            return Lasso()

    def get_vectorized(self):
        if self.name == "bag_of_worlds":
            return CountVectorizer(min_df=0, max_df=1, binary=False, ngram_range=(1, 3))

        return TfidfVectorizer(min_df=0, max_df=1, use_idf=True, ngram_range=(1, 3))

