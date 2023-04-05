import torch
import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, TensorDataset


class VotingBertModel(torch.nn.Module):
    """
    Voting fold ensemble class for robust predictions.
    """
    def __init__(self,
                 checkpoint_list=[],
                 batch_size=64,
                 tokenizer_max_len=16):

        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.tokenizer_max_len = tokenizer_max_len

        if len(checkpoint_list) > 0:
            self.models = torch.nn.ModuleList(
                [AutoModelForSequenceClassification.from_pretrained(checkpoint).to(self.device) \
                 for checkpoint in tqdm(checkpoint_list, desc="Loading models")])
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_list[0])

    def blended_forward(self, x, attention_mask):
        """
        Get probability-voted-model predictions on given input

        ---------
        :param x: List of inference texts
        :param attention_mask: Attention mask got from the tokenizer
        :return: Class-id-ordered prediction probabilities
        """
        probas = torch.zeros((x.shape[0], 5)).to(self.device)

        for model in self.models:
            model.eval()
            raw_output = model(x, attention_mask=attention_mask)
            logits = raw_output.logits
            probas += (torch.nn.functional.softmax(logits, dim=1) / len(self.models))

        return probas

    def predict(self,
                x_test):
        """
        Get single-model predictions on given input

        ---------
        :param x_test: List of inference texts
        :return: Predicted class ids and class-id-ordered prediction probabilities
        """

        test_texts = x_test.to_list()
        test_encodings = self.tokenizer(test_texts,
                                        truncation=True,
                                        padding=True,
                                        max_length=self.tokenizer_max_len)

        test_dataset = TensorDataset(
            torch.tensor(test_encodings['input_ids']),
            torch.tensor(test_encodings['attention_mask']))

        test_loader = DataLoader(test_dataset,
                                 batch_size=self.batch_size * 2,
                                 shuffle=False,
                                 num_workers=0,
                                 pin_memory=True)

        all_preds = []
        all_probas = []
        with torch.no_grad():
            for batch in tqdm(test_loader):
                input_ids, attention_mask = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                probas = self.blended_forward(input_ids,
                                              attention_mask=attention_mask)
                predictions = torch.argmax(probas, dim=1)
                all_preds.extend(predictions.cpu().numpy())
                all_probas.extend(probas.cpu().numpy())
        return all_preds, all_probas

    def set_device(self, device):
        """
        Sets the given device for all inner models.

        ---------
        :param device: Target device('cuda', 'cpu')
        """

        self.device = device
        for m in self.models:
            m.to(device)

    def save(self, path):
        """
        Save the fine-tuned model and its tokenizer to the given path.

        ---------
        :param path: Model output path
        """
        print("Saving VotingBertModel...")
        torch.save(self, path)
        print("Done!")
