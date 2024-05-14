import torch
import os
import pandas as pd
import numpy as np
import random
import string
import nltk
import re
from argparse import ArgumentParser
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from transformers.models.bert import BertConfig , BertModel, BertForSequenceClassification, BertTokenizer
from torch.utils.data import Dataset

nltk.download('stopwords')
nltk.download('gutenberg')
nltk.download('words')
nltk.download('punkt')
nltk.download('wordnet')
word_list = []
random_seed = 42
random.seed(random_seed)

class TextData(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, mode):
        self.mode = mode
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def get_weight(self):
        if self.mode == 'test':
            raise('test mode doesnot support get weight')
        labels = np.array(self.labels)
        counts = np.bincount(labels)
        counts = counts / np.sum(counts)
        return torch.tensor(1 / counts, dtype=torch.float32)
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        if self.mode == "train" or self.mode == "val":
            label = self.labels[idx]
            return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label), 'text': text}
        elif self.mode == "test":
            return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten()}
        else:
            raise("Data reading mode error")
        
def add_mask(text: str):
    text = text.split()
    start_idx = random.randint(0, len(text))
    end_idx = random.randint(start_idx, max(len(text), start_idx + int(len(text) * 0.15)))
    text = text[:start_idx] + ['[MASK]'] + text[end_idx:]
    return " ".join(text)

def clean_text(text: str):
    lemmatizer = nltk.stem.WordNetLemmatizer()

    url_pattern = r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*"
    html_pattern = r"<([a-z]+)(?![^>]*\/>)[^>]*>"
        
    text = re.sub(url_pattern, " ",text)
    text = re.sub(html_pattern, " ", text)
    
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text) #Removing emojis
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    text = nltk.word_tokenize(text.lower())
    processed_text = []
    
    for t in text:
        t = lemmatizer.lemmatize(t)
        processed_text.append(t)
    
    text = " ".join(processed_text) 
    
    return text


def read_data(data_path, tokenizer_root='bert-base-uncased', masked_ratio=0.5, max_length=128, mode='train', analyze=False, type="title_text"):
    assert type in ['title', 'text', 'title_text']
    print(f"Read data with type {type} mode {mode}")
    df = pd.read_json(data_path)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_root, do_lower_case=True)
    text = []
    label = []
    for i, row in tqdm(df.iterrows()):
        t = ""
        if "title" in type:
            t += row['title'] + ' '
        if "text" in type:
            t += row['text'] + ' '
        t = clean_text(t)
        if mode == "train" and random.random() < masked_ratio:
            t = add_mask(t)
        t = '[CLS] ' + t
        text.append(t)
        label.append(row['rating'] - 1 if mode != 'test' else None)
            
    if analyze:
        import matplotlib.pyplot as plt
        text_len = [len(t.split()) for t in text]
        text_len = np.array(text_len)
        text_len.sort()
        for t in text:
            for word in t.split():
                if word not in word_list:
                    word_list.append(word)
        print(len(word_list))
        count = np.unique(text_len, return_index=True, return_counts=True)
        x, h = count[0].tolist(), count[2].tolist()
        print(f'Mean of len: {text_len[int(text_len.size * 0.5)]}')
        print(f'Most of len: {count[0][count[2].argmax()]}')
        print(f'95% of len: {text_len[int(text_len.size * 0.95)]}')
        
        plt.bar(x, h)
        plt.xlabel('text len')
        plt.ylabel('count')
        plt.savefig(f'analyze/text_len.png')
        
    return TextData(text, label if mode != "test" else None, tokenizer, max_length, mode)

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_classes=5, dropout_ratio=0.2, classification_model=False, pretrained_model=True):
        super(BERTClassifier, self).__init__()
        if not pretrained_model:
            self.bert = BertModel(BertConfig(hidden_dropout_prob=dropout_ratio, attention_probs_dropout_prob=dropout_ratio), add_pooling_layer=True)
        elif not classification_model:
            self.bert = BertModel.from_pretrained(bert_model_name)
        else:
            self.bert = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=num_classes)
        self.dropout = nn.Dropout(dropout_ratio)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.classification_model = classification_model

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        if not self.classification_model:
            pooled_output = outputs.pooler_output
            x = self.dropout(pooled_output)
            logits = self.fc(x)
            return logits
        else:
            return outputs.logits

class TrainingAgent():
    def __init__(self):
        train_data_path = "./dataset/train.json"
        val_data_path = "./dataset/val.json"
        test_data_path = "./dataset/test.json"
        parser = ArgumentParser()
        parser.add_argument("--epochs", nargs='?', type=int, default=20)
        parser.add_argument("--batch_size", nargs='?', type=int, default=16)
        parser.add_argument("--lr", nargs='?', type=float, default=2e-5)
        parser.add_argument("--weight_decay", nargs='?', type=float, default=1e-2)
        parser.add_argument("--dropout_ratio", nargs='?', type=float, default=0.4)
        parser.add_argument("--masked_ratio", nargs='?', type=float, default=0.5)
        parser.add_argument("--max_length", nargs='?', type=int, default=256)
        parser.add_argument("--model_root", nargs='?', type=str, default='bert-base-uncased')
        parser.add_argument("--model_save", nargs='?', type=str, default='temp')
        parser.add_argument('--text_type', nargs='?', type=str, default='title_text', help="[title_text, title, text]")
        parser.add_argument("--use_classification", action='store_true', default=False)
        parser.add_argument("--not_use_pretrained", action='store_false', default=True)
        args = parser.parse_args()
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.dropout_ratio = args.dropout_ratio
        self.masked_ratio = args.masked_ratio
        self.max_length = args.max_length
        self.model_root = args.model_root
        self.model_save = args.model_save
        self.text_type = args.text_type
        
        self.model_name = f"bert_classifier_epoch_{self.epochs}_batch_{self.batch_size}_lr_{self.lr}"
        self.model_save_root = os.path.join('model', self.model_root, self.model_save)
        self.prediction_root = os.path.join('predictions', self.model_save)
        self.log_root = os.path.join('logs', f'{self.model_save}')
        os.makedirs(self.model_save_root, exist_ok=True)
        os.makedirs(self.prediction_root, exist_ok=True)
        os.makedirs(self.log_root, exist_ok=True)
        
        self.model = BERTClassifier(self.model_root, 5, self.dropout_ratio, args.use_classification, args.not_use_pretrained).to(self.device)
        train_data = read_data(train_data_path, self.model_root, self.masked_ratio, self.max_length, mode="train", type=self.text_type)
        val_data = read_data(val_data_path, self.model_root, None, self.max_length, mode="val", type=self.text_type)
        test_data = read_data(test_data_path, self.model_root, None, self.max_length, mode="test", type=self.text_type)
        self.train_dataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_data, batch_size=1)
        self.test_dataloader = DataLoader(test_data, batch_size=1)
        self.loss_weight = train_data.get_weight().to(self.device)
        
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        total_steps = len(self.train_dataloader) * self.epochs
        self.scheduler = LinearLR(self.optimizer, total_iters=total_steps)
        self.writer = SummaryWriter(self.log_root)
    
    def train(self):
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            train_acc, train_loss = self._train()
            accuracy, report = self._evaluate()
            print(f"Validation Accuracy: {accuracy:.4f}")
            print(report)
            self.writer.add_scalar('Train/Loss', train_loss, epoch)
            self.writer.add_scalar('Train/Acc', train_acc, epoch)
        torch.save(self.model.state_dict(), os.path.join(self.model_save_root, f"{self.model_name}.pth"))
    
    def inference(self):  
        predictions = self._inference()
        with open(os.path.join(self.prediction_root, f"{self.model_name}.csv"), 'w') as f:
            f.write('index,rating\n')
            for i, pred in enumerate(predictions):
                f.write(f'index_{i},{pred+1}\n')
            
    def _inference(self):
        self.model.load_state_dict(torch.load(os.path.join(self.model_save_root, f"{self.model_name}.pth")))
        self.model.eval()
        predictions = []
        for batch in tqdm(self.test_dataloader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
        return predictions
            
    def _train(self):
        self.model.bert.train()
        self.model.train()
        train_loss = 0
        train_acc = 0
        data_size = len(self.train_dataloader.dataset)
        
        for batch in tqdm(self.train_dataloader):
            self.optimizer.zero_grad()
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss = nn.CrossEntropyLoss(weight=self.loss_weight)(outputs, labels)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            train_loss += loss.item()
            train_acc += (outputs.argmax(dim=-1) == labels).sum().item()
            
        train_loss /= data_size
        train_acc /= data_size
        return train_acc, train_loss
            
    def _evaluate(self):
        self.model.eval()
        predictions = []
        actual_labels = []
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)
                predictions.extend(preds.cpu().tolist())
                actual_labels.extend(labels.cpu().tolist())
        return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)

if __name__ == "__main__":
    agent = TrainingAgent()
    agent.train()
    agent.inference()