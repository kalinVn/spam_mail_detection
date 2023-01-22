import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.toktok import ToktokTokenizer

import re
from bs4 import BeautifulSoup
class PreprocessText:

    def __init__(self, dataset):
        self.dataset = dataset

        self.count_vectorized = CountVectorizer(min_df=0, max_df=1, binary=False, ngram_range=(1, 3))
        self.tf_vectorized = TfidfVectorizer(min_df=0, max_df=1, use_idf=True, ngram_range=(1, 3))

    def get_dataset(self):
        return self.dataset

    def noise_remove_vals(self, text):
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()
        text = re.sub('\[[^]]*\]', '', text)

        return text

    def stemmer(self, text):
        ps = nltk.porter.PorterStemmer()
        text = ' '.join([ps.stem(word) for word in text.split()])

        return text

    def removing_stopwords(self, text):
        nltk.download('stopwords')
        stop_words = nltk.corpus.stopwords.words('english')
        tokenizers = ToktokTokenizer()

        tokens = tokenizers.tokenize(text)
        tokens = [i.strip() for i in tokens]

        fill_tokens = [i for i in tokens if i.lower() not in stop_words]

        filtered_text = ' '.join(fill_tokens)

        return filtered_text




