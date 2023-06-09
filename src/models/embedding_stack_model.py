from sentence_transformers import SentenceTransformer, util
from .bert_model import BertModel
from .base_model import BaseModel


class EmbeddingStackModel(BaseModel):
    def __init__(self,
                 embed_model_path='dbmdz/bert-base-turkish-128k-uncased',
                 head_model=None,
                 head_model_args={},
                 retrain_embed_model=False):
        super().__init__()
        self.embed_model_path = embed_model_path
        self.embedding_model = SentenceTransformer(embed_model_path)
        self.head_model = head_model(**head_model_args)
        self.retrain_embed_model = retrain_embed_model

    def train(self,
              x_train,
              y_train,
              x_val,
              y_val,
              fold_id="none"):
        """
        Run a full training on given training corpus (and evaluate on the validation corpus if given).

        ---------
        :param x_train: train texts
        :param y_train: train classes
        :param x_val: validation texts
        :param y_val: validation classes
        :param fold_id: fold-id identifier for checkpointing
        """

        if self.retrain_embed_model:
            save_path = "./checkpoints/embed_stack_backbone"
            embed_model_obj = BertModel(model_path=self.embed_model_path,
                                        tokenizer_max_len=64,
                                        batch_size=32,
                                        learning_rate=7e-5,
                                        epochs=3,
                                        warmup_ratio=0.1,
                                        mlm_pretrain=False)
            embed_model_obj.train(x_train, y_train, x_val, y_val)
            embed_model_obj.save(save_path)
            self.embedding_model = SentenceTransformer(save_path)
            del embed_model_obj

        x_train = self.embedding_model.encode(x_train.reset_index(drop=True),
                                              convert_to_tensor=False,
                                              batch_size=32)
        self.head_model.fit(x_train, y_train.reset_index(drop=True))

    def predict(self,
                x_test):
        """
        Get model predictions on given input

        ---------
        :param x_test: List of inference texts
        :return: Predicted class ids and class-id-ordered prediction probabilities
        """

        x_test = self.embedding_model.encode(x_test.reset_index(drop=True),
                                             convert_to_tensor=False,
                                             batch_size=32)
        pred = self.head_model.predict(x_test).flatten().tolist()
        pred_proba = self.head_model.predict_proba(x_test)

        return pred, pred_proba
