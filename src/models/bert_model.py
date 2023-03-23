import os
import time

from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import get_cosine_schedule_with_warmup

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:21"

from .base_model import BaseModel

from torch.utils.data import Dataset

import torch
import numpy as np
import os, random

CE_LOSS_OBJ = torch.nn.CrossEntropyLoss(reduction="none")


def adjust_lr(optimizer, epoch):
    lr = 4e-6

    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = 100 * lr

    return lr


def get_optimizer(net):
    params = [x[1] for x in filter(lambda kv: "backbone" in kv[0], net.named_parameters())]
    arc_weight = [x[1] for x in filter(lambda kv: "backbone" not in kv[0], net.named_parameters())]

    optimizer = torch.optim.Adam([{"params": params}, {"params": arc_weight}], lr=4e-6, betas=(0.9, 0.999),
                                 eps=1e-08)
    return optimizer


def ohem_loss(preds, labels):
    losses = CE_LOSS_OBJ(preds, labels)
    top_losses, _ = torch.topk(losses, k=len(losses) // 2)
    return top_losses.mean()


def set_seed(seed=int):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    return random_state


class MLMDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(
            self,
            tokenizer,
            text_list: list,
            block_size: int = 128,
    ):
        batch_encoding = tokenizer(text_list, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


class BertModel(BaseModel):
    def __init__(self,
                 model_path='sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
                 tokenizer_max_len=128,
                 batch_size=16,
                 learning_rate=5e-5,
                 epochs=5,
                 warmup_ratio=0.1,
                 mlm_pretrain=False):

        super().__init__()
        self.model_path = model_path
        self.tokenizer_max_len = tokenizer_max_len
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.warmup_ratio = warmup_ratio
        self.mlm_pretrain = mlm_pretrain

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def save(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)



    def train_mlm(self,
                  x_train):
        set_seed(42)
        model = AutoModelForMaskedLM.from_pretrained(self.model_path)
        tokenizer = AutoTokenizer.from_pretrained(self.model_path,
                                                  ignore_mismatched_sizes=True,
                                                  add_prefix_space=True)
        model.to(self.device)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )

        dataset = MLMDataset(tokenizer=tokenizer,
                             text_list=x_train.to_list(),
                             block_size=self.tokenizer_max_len)

        model_save_path = "./checkpoints/mlm_model"

        training_args = TrainingArguments(
            output_dir=model_save_path,
            save_strategy="no",
            learning_rate=5e-6,
            overwrite_output_dir=True,
            num_train_epochs=8,
            per_gpu_train_batch_size=16,
            prediction_loss_only=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )

        print('Start a trainer...')
        # Start training
        trainer.train()

        # Save
        trainer.save_model(model_save_path)
        tokenizer.save_pretrained(model_save_path)

    def train(self,
              x_train,
              y_train,
              x_val,
              y_val,
              n_batches=1):

        set_seed(42)

        if self.mlm_pretrain:
            self.train_mlm(x_train)
            model_path = "./checkpoints/mlm_model"
        else:
            model_path = self.model_path

        self.model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                                        num_labels=5)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                       ignore_mismatched_sizes=True,
                                                       add_prefix_space=True)

        train_texts = x_train.to_list()
        train_labels = y_train.to_list()
        train_encodings = self.tokenizer(train_texts,
                                         truncation=True,
                                         padding=True,
                                         max_length=self.tokenizer_max_len)

        train_dataset = TensorDataset(
            torch.tensor(train_encodings['input_ids']),
            torch.tensor(train_encodings['attention_mask']),
            torch.tensor(train_labels))

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=False)

        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.learning_rate)

        # optimizer = get_optimizer(self.model)
        scaler = torch.cuda.amp.GradScaler()

        num_train_steps = int(len(train_texts) / self.batch_size * self.epochs)
        num_warmup_steps = num_train_steps * self.warmup_ratio

        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_train_steps)

        for epoch in range(self.epochs):
            self.model.train()

            # adjust_lr(optimizer, epoch)

            running_loss = 0.0
            optimizer.zero_grad()
            for i, batch in tqdm(enumerate(train_loader), desc="Batch", total=len(train_loader)):

                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = ohem_loss(outputs.logits, labels.squeeze(-1))
                    # outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    # loss = outputs.loss

                loss = loss / n_batches
                scaler.scale(loss).backward()

                if ((i + 1) % n_batches) == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                # loss.backward()
                # optimizer.step()

                scheduler.step()
                running_loss += loss.item()

            # Evaluate on validation set
            tr_accuracy, tr_f1 = self.evaluate(x_train, y_train)
            val_accuracy, val_f1 = self.evaluate(x_val, y_val)
            time.sleep(0.5)
            print(
                f'Epoch {epoch + 1}, Training Accuracy: {tr_accuracy:.5f}, Training F1-Macro: {tr_f1:.5f}')
            print(
                f'Epoch {epoch + 1}, Validation Accuracy: {val_accuracy:.5f}, Validation F1-Macro: {val_f1:.5f}')

    def predict(self,
                x_test):

        test_texts = x_test.to_list()  # list of validation texts
        test_encodings = self.tokenizer(test_texts, truncation=True, padding=True, max_length=self.tokenizer_max_len)

        test_dataset = TensorDataset(
            torch.tensor(test_encodings['input_ids']),
            torch.tensor(test_encodings['attention_mask']))

        test_loader = DataLoader(test_dataset,
                                 batch_size=self.batch_size * 2,
                                 shuffle=False,
                                 num_workers=0,
                                 pin_memory=True)

        self.model.eval()

        all_preds = []
        all_probas = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids, attention_mask = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probas = torch.nn.functional.softmax(logits)
                predictions = torch.argmax(logits, dim=1)
                all_preds.extend(predictions.cpu().numpy())
                all_probas.extend(probas.cpu().numpy())

        return all_preds, all_probas

    def evaluate(self, x_val, y_val):
        val_texts = x_val.to_list()  # list of validation texts
        val_labels = y_val.to_list()  # list of validation labels (ints from 0 to num_labels-1)
        val_encodings = self.tokenizer(val_texts, truncation=True, padding=True, max_length=self.tokenizer_max_len)

        val_dataset = TensorDataset(
            torch.tensor(val_encodings['input_ids']),
            torch.tensor(val_encodings['attention_mask']),
            torch.tensor(val_labels))

        val_loader = DataLoader(val_dataset,
                                batch_size=self.batch_size * 2,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True)

        self.model.eval()
        total_correct = 0
        all_preds = []
        all_probas = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probas = torch.nn.functional.softmax(logits)
                predictions = torch.argmax(logits, dim=1)
                total_correct += torch.sum(predictions == labels)
                all_preds.extend(predictions.cpu().numpy())
                all_probas.extend(probas.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        accuracy = total_correct / len(val_loader.dataset)
        f1 = f1_score(all_labels, all_preds, average='macro')
        return accuracy, f1
