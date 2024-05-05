from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
from transformers.models.bert import BertTokenizer
from sklearn.model_selection import train_test_split
import nltk
import re
import random


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
        if self.mode == "train":
            label = self.labels[idx]
            return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}
        elif self.mode == "test":
            return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten()}
        else:
            raise("Data reading mode error")
        
def read_data(data_path, tokenizer_root='bert-base-uncased', max_length=1024, mode='train'):
    df = pd.read_json(data_path)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_root)
    text = []
    label = []
    nltk.download('stopwords')
    nltk_stopwords = nltk.corpus.stopwords.words('english')
    count = 0
    stopword_drop_ratio = 0.5
    verified_drop_ratio = 0.5
    helpful = df['helpful_vote'].to_list()
    verified = df['verified_purchase'].to_list()
    helpful_drop_ratio = 0.3
    for i, row in df.iterrows():
        t = f"[CLS] {row['title']} {row['text']}"
        url_pattern = "((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*"
        t = re.sub(url_pattern, '', t)
        t = t.replace("<br />", " ")
        t_list = t.split()
        t_process = ""
        for w in t_list:
            if w not in nltk_stopwords or random.random() > stopword_drop_ratio:
                t_process += f' {w}'
        if mode == 'test' or (row['verified_purchase'] or random.random() > verified_drop_ratio) or (row['helpful_vote'] > 0 or random.random() > helpful_drop_ratio):
            text.append(t_process)
            label.append(row['rating'] - 1 if mode == 'train' else None)
    if mode == "train":
        train_texts, val_texts, train_labels, val_labels = train_test_split(text, label, test_size=0.2, random_state=42)
        train_data = TextData(train_texts, train_labels, tokenizer, max_length, mode="train")
        val_data = TextData(val_texts, val_labels, tokenizer, max_length, mode="train")
        return train_data, val_data
    elif mode == "test":
        test_data = TextData(text, None, tokenizer, max_length, mode="test")
        return test_data
    else:
        raise("Data reading mode error")
    
if __name__ == '__main__':
    data_path = './dataset/train.json'
    data, _ = read_data(data_path)
    print(data.get_weight())
    
    
    