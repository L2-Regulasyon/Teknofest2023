import torch
import time
import random
import pandas as pd
import os
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from .base_model import BaseModel
from transformers import (AdamW, AutoModelForMaskedLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments, get_cosine_schedule_with_warmup)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:21"


def get_optimizer_grouped_parameters(model,
                                     learning_rate,
                                     weight_decay,
                                     layerwise_learning_rate_decay
                                     ):
    """
    Setting optimizer group paramaters.
    ---------
    param model: Backbone
    param learning_rate: Learning rate
    param weight_decay: Weight decay (L2 penalty)
    param layerwise_learning_rate_decay: layer-wise learning rate decay: a method that applies higher learning rates for top layers and lower learning rates for bottom layers
    return: Optimizer group parameters for training
    """   
    
    model_type = model.config.model_type

    if "roberta" in model.config.model_type:
        model_type = "roberta"

    no_decay = ["bias", "LayerNorm.weight"]
    # initialize lr for task specific layer
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "classifier" in n or "pooler" in n],
            "weight_decay": 0.0,
            "lr": learning_rate,
        },
    ]
    # initialize lrs for every layer
    layers = [getattr(model, model_type).embeddings] + list(getattr(model, model_type).encoder.layer)
    layers.reverse()
    lr = learning_rate
    for layer in layers:
        lr *= layerwise_learning_rate_decay
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr,
            },
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]
    return optimizer_grouped_parameters


def get_llrd_optimizer_scheduler(model,
                                 learning_rate=1e-5,
                                 weight_decay=0.01,
                                 layerwise_learning_rate_decay=0.95,
                                 num_warmup_steps=0,
                                 num_training_steps=10):
    
    """
    Setting optimizer and scheduler paramaters.
    
    ---------
    param model: Backbone
    param learning_rate: Learning rate
    param weight_decay: Weight decay (L2 penalty)
    param layerwise_learning_rate_decay: layer-wise learning rate decay: a method that applies higher learning rates for top layers and lower learning rates for bottom layers
    param num_warmup_steps: warmup steps
    param num_training_steps: Epoch size
    return: Optimizer and scheduler parameters for training
    """
    grouped_optimizer_params = get_optimizer_grouped_parameters(
        model,
        learning_rate, weight_decay,
        layerwise_learning_rate_decay
    )
    optimizer = AdamW(
        grouped_optimizer_params,
        lr=learning_rate,
        correct_bias=True
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    return optimizer, scheduler


def ohem_loss(preds,
              labels,
              weights,
              label_smoothing=0.05):
    """
    OHEM (Online Hard Example Mining) loss for training.
    
    ---------
    param preds: Predicted values.
    param labels: Groundt truth labels.
    param weights: A manual rescaling weight given to each class. 
    param label_smoothing: A float in [0.0, 1.0]. Specifies the amount of smoothing when computing the loss, where 0.0 means no smoothing
    return: Training loss.
    """
    CE_LOSS_OBJ = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).cuda(),
                                            reduction="none",
                                            label_smoothing=label_smoothing
                                            )
    losses = CE_LOSS_OBJ(preds, labels)
    top_losses, _ = torch.topk(losses, k=len(losses) // 2)
    return top_losses.mean()


def set_seed(seed=int):
    """
    Sets the seed for the entire environment so results are the same every time we run. This is for reproducibility.
    
    ---------
    param seed: Seed number
    return: Set seed for torch, numpy, random all over python methods. 
    
    """
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
    Create MLM dataset object
    """

    def __init__(self,
                 tokenizer,
                 text_list: list,
                 block_size: int = 128
                 ):
        batch_encoding = tokenizer(text_list, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


class StratifiedBatchSampler:
    """
    Stratified batch sampling provides equal representation of target classes in each batch
    """

    def __init__(self, y, batch_size, shuffle=True):
        if torch.is_tensor(y):
            y = y.numpy()
        assert len(y.shape) == 1, 'label array must be 1D'
        self.n_batches = int(len(y) / batch_size)
        self.skf = StratifiedKFold(n_splits=self.n_batches, shuffle=shuffle, random_state=1337)
        self.X = torch.randn(len(y), 1).numpy()
        self.y = y
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = torch.randint(0, int(1e8), size=()).item()
        for train_idx, test_idx in self.skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        return self.n_batches


class BertModel(BaseModel):
    """
    Fine-tune the given model for multiclass classification.
    """
    
    def __init__(self,
                 model_path='dbmdz/bert-base-turkish-128k-uncased',
                 auth_token=None,
                 tokenizer_max_len=64,
                 batch_size=32,
                 learning_rate=7e-5,
                 epochs=3,
                 warmup_ratio=0.1,
                 weight_decay=0.01,
                 llrd_decay=0.95,
                 label_smoothing=0.05,
                 grad_clip=1.0,
                 prevent_bias=False,
                 mlm_pretrain=False,
                 mlm_probability=0.15,
                 out_folder=None,
                 experiment_name='',
                 ):
        
        """
        Initiliaze model parameters for traning purpose.
        """

        super().__init__()
        self.out_folder = out_folder
        self.experiment_name = experiment_name
        self.model_path = model_path
        self.auth_token = auth_token
        self.tokenizer_max_len = tokenizer_max_len
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.llrd_decay = llrd_decay
        self.label_smoothing = label_smoothing
        self.grad_clip = grad_clip
        self.prevent_bias = prevent_bias
        self.mlm_pretrain = mlm_pretrain
        self.mlm_probability = mlm_probability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load(self):
        """
        Load model and tokenizer, send it to the device 
        """
        
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path,
                                                                        use_auth_token=self.auth_token,
                                                                        num_labels=5)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path,
                                                       use_auth_token=self.auth_token,
                                                       ignore_mismatched_sizes=True,
                                                       add_prefix_space=True)

    def save(self,
             path: str):
        """
        Save fine-tuned model to the given path.
        
        ---------
        param path: Model output path
        return: Save trained model
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def train_mlm(self,
                  x_train):
        
        """
        Run a MaskedLM pre-training on given training corpus.
        
        ---------
        param x_train: train dataset
        return: Save the trained model.
        """
        
        set_seed(42)
        model = AutoModelForMaskedLM.from_pretrained(self.model_path)
        tokenizer = AutoTokenizer.from_pretrained(self.model_path,
                                                  ignore_mismatched_sizes=True,
                                                  add_prefix_space=True)
        model.to(self.device)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=self.mlm_probability
        )

        dataset = MLMDataset(tokenizer=tokenizer,
                             text_list=x_train.to_list(),
                             block_size=self.tokenizer_max_len)

        model_save_path = "./checkpoints/mlm_model"

        training_args = TrainingArguments(
            output_dir=model_save_path,
            save_strategy="no",
            learning_rate=1e-5,
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_gpu_train_batch_size=16,
            prediction_loss_only=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )

        # Start training
        print('Start a trainer...')
        trainer.train()

        # Save
        trainer.save_model(model_save_path)
        tokenizer.save_pretrained(model_save_path)

    def train(self,
              x_train,
              y_train,
              x_val=0,
              y_val=0,
              n_batches=1,
              fold_id="none"):

        set_seed(42)

        # Class Weighting
        cls_weights = list(dict(sorted(dict(1 / ((y_train.value_counts(normalize=True)) ** (1 / 3))).items())).values())
        cls_weights /= min(cls_weights)
        print("Class weights:", cls_weights)

        # MLM Pretraining
        if self.mlm_pretrain:
            self.train_mlm(x_train)
            model_path = "./checkpoints/mlm_model"
        else:
            model_path = self.model_path

        # Model initialization
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                                        num_labels=5,
                                                                        ignore_mismatched_sizes=True)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                       ignore_mismatched_sizes=True,
                                                       add_prefix_space=True)

        # Unbiasing the data
        if self.prevent_bias == 2:
            # Length clipping for 'OTHER' class
            offensive_lens = x_train[y_train != 0].str.split().apply(len).sample(
                len(x_train[y_train == 0]), replace=True
            ).values

            k = pd.DataFrame(x_train[y_train == 0].str.split()).reset_index(drop=True).reset_index()
            k = k.apply(
                lambda x: " ".join(x.text[:offensive_lens[x["index"]]]), axis=1).values

            x_train.loc[y_train == 0] = k

        # Dataset creation
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

        # Dataloader creation
        train_loader = DataLoader(train_dataset,
                                  batch_sampler=StratifiedBatchSampler(y_train,
                                                                       batch_size=self.batch_size,
                                                                       shuffle=True),
                                  num_workers=0,
                                  pin_memory=False,
                                  )

        # Optimizator, LLRD and scheduler
        num_train_steps = int(len(train_texts) / self.batch_size * self.epochs)
        num_warmup_steps = int(num_train_steps * self.warmup_ratio)

        optimizer, scheduler = get_llrd_optimizer_scheduler(self.model,
                                                            learning_rate=self.learning_rate,
                                                            weight_decay=self.weight_decay,
                                                            layerwise_learning_rate_decay=self.llrd_decay,
                                                            num_warmup_steps=num_warmup_steps,
                                                            num_training_steps=num_train_steps)

        scaler = torch.cuda.amp.GradScaler()

        # Training loop
        for epoch in range(self.epochs):
            self.model.train()

            running_loss = 0.0
            optimizer.zero_grad()
            for i, batch in tqdm(enumerate(train_loader), desc="Batch", total=len(train_loader)):

                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = ohem_loss(outputs.logits,
                                     labels.squeeze(-1),
                                     weights=cls_weights,
                                     label_smoothing=self.label_smoothing)

                loss = loss / n_batches
                scaler.scale(loss).backward()

                if ((i + 1) % n_batches) == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                scheduler.step()
                running_loss += loss.item()

            # Evaluate on validation set
            tr_accuracy, tr_f1 = self.evaluate(x_train, y_train)
            if isinstance(x_val, pd.Series) and isinstance(y_val, pd.Series):
                val_accuracy, val_f1 = self.evaluate(x_val, y_val)
            time.sleep(0.5)
            print(
                f'Epoch {epoch + 1}, Training Accuracy: {tr_accuracy:.5f}, Training F1-Macro: {tr_f1:.5f}')
            if isinstance(x_val, pd.Series) and isinstance(y_val, pd.Series):
                print(
                    f'Epoch {epoch + 1}, Validation Accuracy: {val_accuracy:.5f}, Validation F1-Macro: {val_f1:.5f}')

        # Saving checkpoint
        if self.out_folder:
            fold_id = "_" + fold_id if (fold_id != "none") else ""
            self.save(f"{self.out_folder}/{self.experiment_name}{fold_id}")

    def predict(self,
                x_test,
                progress=False):
        """
        Get model predictions on given input

        ---------
        param x_test: List of inference texts
        return: Predicted class ids and class-id-ordered prediction probabilities
        """
        test_texts = x_test.to_list()  # list of validation texts
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

        self.model.eval()

        all_preds = []
        all_probas = []

        if progress:
            test_loader = tqdm(test_loader)

        with torch.no_grad():
            for batch in test_loader:
                input_ids, attention_mask = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probas = torch.nn.functional.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                all_preds.extend(predictions.cpu().numpy())
                all_probas.extend(probas.cpu().numpy())

        return all_preds, all_probas

    def evaluate(self,
                 x_val,
                 y_val):
        
        """
        Evaluate the model w.r.t. given ground truth.
        
        ---------
        param x_val: List of validation texts
        param y_val: List of validation labels 
        return: F1 macro and accuracy scores to evaluate the results.
        """
        
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
