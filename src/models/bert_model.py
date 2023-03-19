from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM
from transformers import get_cosine_schedule_with_warmup, DataCollatorWithPadding

import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

import time

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:21"

from .base_model import BaseModel


class BertModel(BaseModel):
    def __init__(self,
                 model_path='sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
                 tokenizer_max_len=128,
                 batch_size=16,
                 learning_rate=5e-5,
                 epochs=5,
                 warmup_ratio=0.1):

        super().__init__()

        self.tokenizer_max_len = tokenizer_max_len
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.warmup_ratio = warmup_ratio

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                                        ignore_mismatched_sizes=True,
                                                                        num_labels=5)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                       ignore_mismatched_sizes=True)

    def train(self,
              x_train,
              y_train,
              x_val,
              y_val):

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

        self.model.train()
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=False)

        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.learning_rate)

        num_train_steps = int(len(train_texts) / self.batch_size * self.epochs)
        num_warmup_steps = num_train_steps * self.warmup_ratio
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)

        total_epochs = []
        epoch_val_score = []
        for epoch in range(self.epochs):
            total_epochs.append(epoch)
            running_loss = 0.0
            for i, batch in tqdm(enumerate(train_loader), desc="Batch", total=len(train_loader)):
                optimizer.zero_grad()
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                running_loss += loss.item()
                # if i % 50 == 49:  # Print every 50 batches
                #     print(f'Epoch {epoch + 1}, Batch {i + 1}/{len(train_loader)}, Loss: {running_loss / 50:.4f}')
                #     running_loss = 0.0
            
            # Evaluate on validation set
            val_accuracy, val_f1 = self.evaluate(x_val, y_val)
            epoch_val_score.append(val_f1)
            time.sleep(0.5)
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
                                batch_size=self.batch_size*2,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True)

        self.model.eval()

        all_preds = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids, attention_mask = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)
                all_preds.extend(predictions.cpu().numpy())

        return all_preds

    def evaluate(self, x_val, y_val):
        val_texts = x_val.to_list()  # list of validation texts
        val_labels = y_val.to_list()  # list of validation labels (ints from 0 to num_labels-1)
        val_encodings = self.tokenizer(val_texts, truncation=True, padding=True, max_length=self.tokenizer_max_len)

        val_dataset = TensorDataset(
            torch.tensor(val_encodings['input_ids']),
            torch.tensor(val_encodings['attention_mask']),
            torch.tensor(val_labels))

        val_loader = DataLoader(val_dataset,
                                batch_size=self.batch_size*2,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True)

        self.model.eval()
        total_correct = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)
                total_correct += torch.sum(predictions == labels)
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        accuracy = total_correct / len(val_loader.dataset)
        f1 = f1_score(all_labels, all_preds, average='macro')
        return accuracy, f1
    
 #%%   
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        mlm_inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', max_length=self.max_len, truncation=True)
        mlm_input_ids = mlm_inputs['input_ids'].squeeze()
        mlm_labels = mlm_input_ids.clone()
        mlm_labels[mlm_labels == self.tokenizer.pad_token_id] = -100 # ignore padding tokens for MLM loss
        
        inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', max_length=self.max_len, truncation=True)
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label,
            'mlm_input_ids': mlm_input_ids,
            'mlm_labels': mlm_labels
        }
    
class BertModelwithMaskedLM(BaseModel):
    def __init__(self,
                 model_path='dbmdz/bert-base-turkish-128k-uncased',
                 tokenizer_max_len=40,
                 batch_size=32,
                 learning_rate=5e-5,
                 epochs=1,
                 warmup_ratio=0.1):

        super().__init__()

        self.tokenizer_max_len = tokenizer_max_len
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.warmup_ratio = warmup_ratio

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                                        #ignore_mismatched_sizes=True,
                                                                        num_labels=5)
        self.mlm_model = AutoModelForMaskedLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.mlm_model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                       #ignore_mismatched_sizes=True
                                                       )

    def train(self,
              x_train,
              y_train,
              x_val,
              y_val):

       #X_train = X_train.to_list()
       #X_val = X_val.to_list()

       #y_train = y_train.to_list()
       #y_val = y_val.to_list()     
        
        train_dataset = CustomDataset(x_train.to_list(),
                                      y_train.to_list(),
                                      self.tokenizer,
                                      self.tokenizer_max_len)
        valid_dataset = CustomDataset(x_val.to_list(),
                                      y_val.to_list(),
                                      self.tokenizer,
                                      self.tokenizer_max_len)
        
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      pin_memory=False)
        valid_dataloader = DataLoader(valid_dataset,
                                      batch_size=self.batch_size*2,
                                      shuffle=False,
                                      pin_memory=False)

        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.learning_rate)

        num_train_steps = int(len(x_train) / self.batch_size * self.epochs)
        num_warmup_steps = num_train_steps * self.warmup_ratio
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)

        total_epochs = []
        epoch_val_score = []
        for epoch in range(self.epochs):
            self.model.train()
            total_epochs.append(epoch)
            train_preds = []
            train_labels = []
            train_loss = 0.0
            for i, batch in tqdm(enumerate(train_dataloader), desc="Batch", total=len(train_dataloader)):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                #MLM
                mlm_input_ids = batch['mlm_input_ids'].to(self.device)
                mlm_outputs = self.mlm_model(mlm_input_ids, labels=mlm_input_ids)
                mlm_loss = mlm_outputs[0]
                
                #Classifier
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                
                # Backward
                loss = outputs[0] + mlm_loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                preds = outputs[1].detach().cpu().numpy()
                labels = labels.to('cpu').numpy()
                train_preds.append(preds)
                train_labels.append(labels)
                train_loss += loss.item()
                # if i % 50 == 49:  # Print every 50 batches
                #     print(f'Epoch {epoch + 1}, Batch {i + 1}/{len(train_loader)}, Loss: {train_loss / 50:.4f}')
                #     train_loss = 0.0
            
            # Evaluate on validation set
            val_loss, val_f1 = self.evaluate(valid_dataloader)
            epoch_val_score.append(val_f1)
            time.sleep(0.5)
            print(
                f'Epoch {epoch + 1}, Validation Loss: {val_loss:.5f}, Validation F1-Macro: {val_f1:.5f}')

    def predict(self,
                x_test):
        
        test_dataset = CustomDataset(x_test.to_list(),
                                     np.zeros(len(x_test.to_list())),
                                     self.tokenizer,
                                     self.tokenizer_max_len)


        test_dataloader = DataLoader(test_dataset,
                                      batch_size=self.batch_size*2,
                                      shuffle=False,
                                      pin_memory=False)

        self.model.eval()

        all_preds = []
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)
                all_preds.extend(predictions.cpu().numpy())

        return all_preds

    def evaluate(self, dataloader):

        self.model.eval()
        total_correct = 0
        valid_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                
                loss = outputs[0]
                valid_loss += loss.item()
                preds = outputs[1].detach().cpu().numpy()
                labels = labels.to('cpu').numpy()
                all_preds.append(preds)
                all_labels.append(labels)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        valid_f1 = f1_score(all_labels, np.argmax(all_preds, axis=1), average='macro')
        valid_loss = valid_loss / len(dataloader)
        
        return valid_loss, valid_f1
