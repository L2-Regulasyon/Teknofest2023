import scipy
import fasttext
from tqdm import tqdm
from .base_model import BaseModel


class VectorStackModel(BaseModel):
    """
    Training and inferece class for Stage 2. This class can bu used for lgbm, xgb, catboost, fasttext, tfidf models.
    """
    def __init__(self,
                 vector_model=None,
                 vector_model_args={},
                 head_model=None,
                 head_model_args={}):
        super().__init__()
        if vector_model != "fasttext":
            self.vector_model = vector_model(**vector_model_args)
        else:
            self.vector_model = "fasttext"
        self.head_model = head_model(**head_model_args)

    def train(self,
              x_train,
              y_train):
        """
        Train vector and head (classifier) model.
    
        ---------
        :param x_train: Train embeddings
        :param y_train: Target
        :return: Trained model
        """

        if self.vector_model == "fasttext":
            with open('temp_fasttext_train.txt', 'w', encoding='utf-8') as f:
                for line in x_train.values:
                    f.write(line)
                    f.write('\n')
            self.vector_model = fasttext.train_unsupervised("temp_fasttext_train.txt")
            self.vector_model.transform = lambda x: scipy.sparse.csr_matrix([self.vector_model.get_sentence_vector(elm.replace("\n", "")) for elm in tqdm(x.values.tolist(),
                                                                                                             desc="FastText")])
            x_train = self.vector_model.transform(x_train)
        else:
            x_train = self.vector_model.fit_transform(x_train)

        self.head_model.fit(x_train, y_train)

    def predict(self,
                x_test):
        """
        Inference on new data.
        
        :param x_test: Test embeddings
        :return: Predicted class ids and class-id-ordered prediction probabilities
        """
        x_test = self.vector_model.transform(x_test)
        pred = self.head_model.predict(x_test).flatten().tolist()
        pred_proba = self.head_model.predict_proba(x_test)
        return pred, pred_proba
