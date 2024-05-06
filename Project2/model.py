import torch
import os
from argparse import ArgumentParser
from torch import nn
from transformers.models.bert import BertModel, BertForSequenceClassification
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from data import read_data
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes, dropout_ratio, classification_model=False):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name) if not classification_model \
            else BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=num_classes)
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
        test_data_path = "./dataset/test.json"
        parser = ArgumentParser()
        parser.add_argument("--epochs", nargs='?', type=int, default=20)
        parser.add_argument("--batch_size", nargs='?', type=int, default=8)
        parser.add_argument("--lr", nargs='?', type=float, default=2e-5)
        parser.add_argument("--weight_decay", nargs='?', type=float, default=1e-2)
        parser.add_argument("--dropout_ratio", nargs='?', type=float, default=0.2)
        parser.add_argument("--max_length", nargs='?', type=int, default=128)
        parser.add_argument("--model_root", nargs='?', type=str, default='bert-base-uncased')
        parser.add_argument("--model_save", nargs='?', type=str, default='temp')
        parser.add_argument("--use_classification", action='store_true', default=False)
        args = parser.parse_args()
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.dropout_ratio = args.dropout_ratio
        self.max_length = args.max_length
        self.model_root = args.model_root
        self.model_save = args.model_save
        
        self.model_name = f"bert_classifier_epoch_{self.epochs}_batch_{self.batch_size}_lr_{self.lr}"
        self.model_save_root = os.path.join('model', self.model_root, self.model_save)
        self.prediction_root = os.path.join('predictions', self.model_save)
        self.log_root = os.path.join('logs', self.model_save, self.model_name)
        os.makedirs(self.model_save_root, exist_ok=True)
        os.makedirs(self.prediction_root, exist_ok=True)
        os.makedirs(self.log_root, exist_ok=True)
        
        self.model = BERTClassifier(self.model_root, 5, self.dropout_ratio, args.use_classification).to(self.device)
        train_data, val_data = read_data(train_data_path, self.model_root, self.max_length, mode="train")
        test_data = read_data(test_data_path, self.model_root, self.max_length, mode="test")
        self.train_dataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_data, batch_size=self.batch_size)
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
            for batch in self.val_dataloader:
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