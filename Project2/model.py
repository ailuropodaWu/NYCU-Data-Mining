from torch import nn
from transformers.models.bert import BertConfig , BertModel, BertForSequenceClassification

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
        
if __name__ == '__main__':
    model = BERTClassifier()
    for module in model.modules():
        print(module)