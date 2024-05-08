from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
from transformers.models.bert import BertTokenizer
from sklearn.model_selection import train_test_split
import nltk
import re
import random
from argparse import ArgumentParser

nltk.download('stopwords')
parser = ArgumentParser()
parser.add_argument("--text_type", nargs='?', type=str, default='title_comment', help="[title_comment, title, comment]")
TYPE = parser.parse_args().text_type

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
    stopword_drop_ratio = 0.5
    nltk_stopwords = nltk.corpus.stopwords.words('english')
    url_pattern = r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*"
    punctuations_pattern = r"[,.;@#?!&$]+\ *"
    html_pattern = r"<([a-z]+)(?![^>]*\/>)[^>]*>"
        
    text = text.lower()
    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)
    text = re.sub(url_pattern, " ",text) 
    text = re.sub(punctuations_pattern, " ",text) 
    text = re.sub(html_pattern, " ", text)
        
    text = [word.lower() for word in text.split() if word.lower() not in nltk_stopwords or random.random() > stopword_drop_ratio]
    
    text = " ".join(text) 
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text) #Removing emojis
    
    return text

def read_data(data_path, tokenizer_root='bert-base-uncased', max_length=128, mode='train', analyze=False):
    assert TYPE in ['title', 'comment', 'title_comment']
    print(f"Read data with type {TYPE} mode {mode}")
    df = pd.read_json(data_path)
    random_seed = 42
    random.seed(random_seed)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_root, do_lower_case=True)
    text = []
    label = []
    verified_drop_ratio = 0.5
    helpful_drop_ratio = 0.3
    masked_ratio = 0.5
    for i, row in df.iterrows():
        t = ""
        if "title" in TYPE:
            t += row['title'] + ' '
        if "comment" in TYPE:
            t += row['text'] + ' '
            
        if mode != 'train' or \
            (row['verified_purchase'] or random.random() > verified_drop_ratio) or \
            (row['helpful_vote'] > 0 or random.random() > helpful_drop_ratio):
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
        count = np.unique(text_len, return_index=True, return_counts=True)
        x, h = count[0].tolist(), count[2].tolist()
        print(f'Mean of len: {text_len[int(text_len.size * 0.5)]}')
        print(f'Most of len: {count[0][count[2].argmax()]}')
        print(f'Quot of len: {text_len[int(text_len.size * 0.95)]}')
        
        plt.bar(x, h)
        plt.savefig('analyze/text_len.png')
        
    return TextData(text, label if mode != "test" else None, tokenizer, max_length, mode)
    
if __name__ == '__main__':
    data_path = './dataset/train.json'
    data = read_data(data_path, mode='train')
    print(data[0])
    print(data[1])
    
    data_path = './dataset/val.json'
    data = read_data(data_path, mode='val')
    print(data[0])
    print(data[1])
    
    
    