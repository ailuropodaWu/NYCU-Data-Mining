from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
import torch.nn as nn

# BERT Model
class BertClassifier(BertPreTrainedModel):
    def __init__(self, config, args):
        super(BertClassifier, self).__init__(config)
        self.bert = BertModel(config)
        self.num_labels = args["num_class"]
        self.dropout = nn.Dropout(args["dropout"])
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    # forward function, data in model will do this
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, labels=None, output_attentions=None,
                output_hidden_states=None, return_dict=None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # bert output
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        # get its [CLS] logits
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output) # add dropout
        logits = self.classifier(pooled_output) # add linear classifier

        return logits