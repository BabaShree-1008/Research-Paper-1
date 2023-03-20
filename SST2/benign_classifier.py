import os

import mlflow
import numpy as np
from datasets import load_metric
from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import logging

logging.set_verbosity_info()

os.environ["MLFLOW_FLATTEN_PARAMS"] = "1"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "sst2"
os.environ["MLFLOW_TAGS"] = '{"classification_type": "benign"}'


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForMaskedLM.from_pretrained('roberta-base')

train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    block_size=512,
    file_path="./dataset/benign/train_text.txt"
)

val_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    block_size=512,
    file_path="./dataset/benign/validation_text.txt"
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
metric = load_metric("accuracy")


def compute_accuracy(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./training_output",
    overwrite_output_dir=True,
    num_train_epochs=25,
    per_device_train_batch_size=48,
    save_steps=500,
    save_total_limit=2,
    seed=1, 
    per_gpu_train_batch_size=8, 
    per_gpu_eval_batch_size=8, 
    gradient_checkpointing=False, 
    no_cuda=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset, 
    eval_dataset=val_dataset, 
    compute_metrics=compute_accuracy
    )

trainer.save_model("./sst2_model")

mlflow.end_run()
