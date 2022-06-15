import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from transformers import TrainingArguments
from datasets import Dataset, DatasetDict


def is_empty_line(line: str) -> bool:
    empty_line = line.strip() == ""
    if empty_line:
        return True
    return False


def is_divider(
    line: str, divider_string: str, empty_line_as_dcoument_divider: bool
) -> bool:

    if is_empty_line(line) and empty_line_as_dcoument_divider:
        return True
    else:
        first_token = line.split()[0]
        if first_token == divider_string:
            return True
        else:
            return False


def load_conll_format_datasets(
    datafiles: Dict,
    column_names: List[str],
    docstart_string: str,
    empty_line_as_dcoument_divider: bool,
) -> DatasetDict:
    raw_datasets = DatasetDict()
    for file in datafiles:
        current_dataset = {}
        for name in column_names:
            current_dataset[name] = []

        current_ex = [[] for _ in range(len(column_names))]
        with open(datafiles[file], "r") as fin:
            for line in fin:
                if is_divider(
                    line,
                    divider_string=docstart_string,
                    empty_line_as_dcoument_divider=empty_line_as_dcoument_divider,
                ):
                    if len(current_ex[0]) > 0:
                        # Append current example in the datsets
                        for i, name in enumerate(column_names):
                            current_dataset[name].append(current_ex[i])
                        current_ex = [[] for _ in range(len(column_names))]
                elif is_empty_line(line):
                    continue
                else:
                    values = line.split()
                    assert len(values) == len(column_names)
                    for i in range(len(column_names)):
                        current_ex[i].append(values[i])
        current_dataset = Dataset.from_dict(current_dataset)

        raw_datasets[file] = current_dataset

    print(raw_datasets)
    return raw_datasets


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    use_crf: bool = field(
        default=True,
        metadata={"help": ("use crf layer on top of classification layer")},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={
            "help": "Will enable to load a pretrained model whose head dimensions are different."
        },
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(
        default="ner", metadata={"help": "The name of the task (ner, pos...)."}
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a csv or JSON file)."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to predict on (a csv or JSON file)."
        },
    )
    conll_format_column_names: Optional[Tuple[str]] = field(
        default=("words", "pos-tags", "chunk-tags", "ner-tags"),
        metadata={
            "help": "names that should be given to each columns of conll format datasets"
        },
    )

    conll_format_docstart_string: Optional[str] = field(
        default="-DOCSTART-",
        metadata={"help": "string indicating start of new document."},
    )

    conll_format_empty_line_as_dcoument_divider: Optional[bool] = field(
        default=True,
        metadata={
            "help": "weather to treat an empty line as document divider or not. across empty line documents to be considered as different examples."
        },
    )

    text_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The column name of text to input in the file (a csv or JSON file)."
        },
    )
    label_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The column name of label to input in the file (a csv or JSON file)."
        },
    )

    label_encoding: Optional[str] = field(
        default="BIO",
        metadata={
            "help": 'Indicates label encoding to find contraint transition. Current choices are "BIO", "IOB1", "BIOUL", and "BMES".'
        },
    )

    include_start_end_transitions: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to include the start and end transition parameters in CRF contraints."
        },
    )

    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. If set, sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    label_all_tokens: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to put the label for one word on all tokens of generated by that word or just on the "
                "one (in which case the other tokens will have a padding index)."
            )
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={
            "help": "Whether to return all the entity levels during evaluation or just the overall ones."
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                ], "`train_file` should be a csv, txt or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                ], "`validation_file` should be a csv, txt or a json file."
        self.task_name = self.task_name.lower()
