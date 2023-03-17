from lightgbm import LGBMClassifier
from sentence_transformers import SentenceTransformer, util

from .base_model import BaseModel


class EmbeddingGBDTModel(BaseModel):
    def __init__(self,
                 model_path='sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
                 batch_size=16):
        super().__init__()
        self.embedding_model = SentenceTransformer(model_path)
        self.gbdt_model = LGBMClassifier()
        self.batch_size = batch_size

    def train(self,
              x_train,
              y_train):
        x_train = self.embedding_model.encode(x_train.reset_index(drop=True),
                                              convert_to_tensor=False,
                                              batch_size=self.batch_size)
        self.gbdt_model.fit(x_train, y_train.reset_index(drop=True))

    def predict(self,
                x_test):
        x_test = self.embedding_model.encode(x_test.reset_index(drop=True),
                                             convert_to_tensor=False,
                                             batch_size=self.batch_size)
        return self.gbdt_model.predict(x_test)
