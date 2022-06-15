# all token classification model with crf head
from transformers import (
    AutoModel,
    AutoModelForTokenClassification,
)
from transformers.modeling_outputs import TokenClassifierOutput
from .crf import ConditionalRandomField


class AutoCrfModelforSequenceTagging(AutoModelForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.base_model = AutoModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = ConditionalRandomField(
            self.num_labels,
            label_encoding=self.config.label_encoding,  # TODO
            idx2tag=self.config.id2label,  # TODO
            include_start_end_transitions=self.config.include_start_end_transitions,  # TODO
        )
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        head_mask=None,
        token_type_ids=None,
        inputs_embeds=None,
        labels=None,
        output_hidden_states=None,
        output_attentions=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.base_model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss = -self.crf(logits, labels, attention_mask)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def freeze_till_clf(self):
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.dropout.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = False

    def freeze_encoder_layer(self):
        for param in self.bert.parameters():
            param.requires_grad = False
