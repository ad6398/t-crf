# all token classification model with crf head
from transformers import AutoModel, AutoModelForTokenClassification, PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
from .crf import ConditionalRandomField
import logging

from torch import nn


class CRFforSequenceTagging(PreTrainedModel):
    def __init__(self, model_args, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        clf_model_class = (
            AutoModelForTokenClassification
            if model_args.custom_clf_model_class is None
            else model_args.custom_clf_model_class
        )
        self.clf_model = clf_model_class.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )
        print("config from initalizer ,", self.config)

        self.crf = ConditionalRandomField(
            self.num_labels,
            label_encoding=self.config.label_encoding,  # TODO
            id2label=self.config.id2label,  # TODO
            label2id=config.label2id,
            include_start_end_transitions=self.config.include_start_end_transitions,  # TODO
        )
        # self.init_weights()

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
        outputs = self.clf_model(
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
        logits = outputs[0]

        loss = None
        if labels is not None:
            loss = -self.crf(logits, labels, attention_mask)
        best_path = self.crf.viterbi_tags(logits, mask=attention_mask)
        # ignore score of path, just store the tags value
        best_path = [x for x, _ in best_path]
        logits *= 0
        for i, path in enumerate(best_path):
            for j, tag in enumerate(path):
                # logits value for each tag
                logits[i, j, int(tag)] = 1

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class AutoCrfModelforSequenceTagging(AutoModelForTokenClassification):
    def __init__(self, config):
        # super().__init__(config)
        self.num_labels = config.num_labels
        self.base_model = AutoModel(config)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        print("config from initalizer ,", self.config)
        logging.info("config from initalizer {}".format(self.config))
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
        sequence_output = self.dropout1(sequence_output)
        sequence_output = self.dropout2(sequence_output)
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
