from torch.utils.data import Dataset
import pandas as pd
import torch
from transformers.models.bert import BertTokenizer
from sklearn.model_selection import train_test_split



class TextData(Dataset):

    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}
    
def read_data(data_path, tokenizer_root='bert-base-uncased', max_length=1024):
    df = pd.read_json(data_path)
    text = []
    for i, row in df.iterrows():
        t = f"Title: {row['title']}. Content: {row['text']}."
        text.append(t.replace("<br />", " "))
    helpful = df['helpful_vote'].to_list()
    verified = df['verified_purchase'].to_list()
    label = [l - 1 for l in df['rating'].to_list()]
    tokenizer = BertTokenizer.from_pretrained(tokenizer_root)
    train_texts, val_texts, train_labels, val_labels = train_test_split(text, label, test_size=0.2, random_state=42)

    train_data = TextData(train_texts, train_labels, tokenizer, max_length)
    val_data = TextData(val_texts, val_labels, tokenizer, max_length)
    return train_data, val_data
    
if __name__ == '__main__':
    data_path = './dataset/train.json'
    data = read_data(data_path)
    print(len(data))
    print(data[0])
    
    
    