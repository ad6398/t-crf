# all token classification model with crf head
from transformers import AutoModel, AutoModelForTokenClassification, PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
from .crf import ConditionalRandomField
import logging

from torch import nn


class CRFforSequenceTagging(PreTrainedModel):
    def __init__(self, config, model_args):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        clf_model_class = (
            AutoModelForTokenClassification
            if model_args.custom_clf_model_class is None
            else model_args.custom_clf_model_class
        )
        self.clf_model = clf_model_class.from_config(config)

        self.crf = ConditionalRandomField(
            self.num_labels,
            label_encoding=self.config.label_encoding,  # TODO
            id2label=self.config.id2label,  # TODO
            label2id=config.label2id,
            include_start_end_transitions=self.config.include_start_end_transitions,  # TODO
        )
        self.post_init()

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

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
