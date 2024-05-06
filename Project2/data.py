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
            return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label), 'text': text}
        elif self.mode == "test":
            return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten()}
        else:
            raise("Data reading mode error")
        
def read_data(data_path, tokenizer_root='bert-base-uncased', max_length=128, mode='train', analyze=False):
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
        t = f"{row['title']} {row['text']}"
        url_pattern = r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*"
        clean_pattern = r"[,.;@#?!&$]+\ *"

        t = t.replace("<br />", " ")
        t = re.sub(url_pattern, '', t)
        t = re.sub(clean_pattern, '', t)
        t = t.lower()
        t = "[CLS] " + t
        t_list = t.split()
        t_process = ""
        for w in t_list:
            if w not in nltk_stopwords or random.random() > stopword_drop_ratio:
                t_process += f' {w}'    
        if mode == 'test' or (row['verified_purchase'] or random.random() > verified_drop_ratio) or (row['helpful_vote'] > 0 or random.random() > helpful_drop_ratio):
            text.append(t_process)
            label.append(row['rating'] - 1 if mode == 'train' else None)
    if analyze:
        import matplotlib.pyplot as plt
        text_len = [len(t.split()) for t in text]
        text_len = np.array(text_len)
        text_len.sort()
        count = np.unique(text_len, return_index=True, return_counts=True)
        x, h = count[0].tolist(), count[2].tolist()
        print(f'Mean of len: {text_len[int(text_len.size * 0.5)]}')
        print(f'Most of len: {count[0][count[2].argmax()]}')
        print(f'Quot of len: {text_len[int(text_len.size * 0.95)]}')
        
        plt.bar(x, h)
        plt.savefig('analyze/text_len.png')
        
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
    data, val_data = read_data(data_path, mode='train')
    print(data[0])
    print(data[1])
    
    
    