from .base_model import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier


class TfidfModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.gbdt_model = LGBMClassifier()
        self.tfidf = TfidfVectorizer()

    def train(self,
              x_train,
              y_train):
        x_train = self.tfidf.fit_transform(x_train)
        self.gbdt_model.fit(x_train, y_train)

    def predict(self,
                x_test):
        x_test = self.tfidf.transform(x_test)
        return self.gbdt_model.predict(x_test)