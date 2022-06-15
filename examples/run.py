from tcrf.main import main, DataArguments, ModelArguments, TrainingArguments

model_args = ModelArguments(
    model_name_or_path="bert-base-uncased",
    use_crf=True,
)
data_args = DataArguments(
    train_file="../data/bc5cdr/train.txt",
    validation_file="../data/bc5cdr/dev.txt",
    test_file="../data/bc5cdr/test.txt",
    # pad_to_max_length=True,
    overwrite_cache=True,
    preprocessing_num_workers=8,
    return_entity_level_metrics=True,
    # conll_format_column_names= [],
    text_column_name="words",
    label_column_name="ner-tags",
    label_encoding="BIOUL",
)

train_args = TrainingArguments(
    output_dir=f"/media/nas_mount/Debanjan/amardeep/tcrf/conll",
    learning_rate=4e-5,
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    do_train=True,
    do_eval=True,
    do_predict=True,
    evaluation_strategy="steps",
    save_steps=500,
    eval_steps=500,
    logging_steps=500,
    # disable_tqdm=True,
)

main(model_args, data_args, train_args)
