from transformers import AutoTokenizer, LlamaForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, BitsAndBytesConfig
from datasets import load_dataset, DatasetDict
from collections import Counter
import evaluate
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import torch

lora_config = LoraConfig(
    r=32,
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"],    # lora
    modules_to_save=["score"]   # else to train and save without lora
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)


# 2. datasets -> datasetDict
train_set = load_dataset('nyu-mll/multi_nli', split='train').select_columns(["premise", "hypothesis", "label"])
train_split = train_set.train_test_split(test_size=0.003)
train_set = train_split["train"]
val_set = train_split["test"]

# print(train_set['label']) list of ints

datasets = DatasetDict({
    "train": train_set,
    "validation": val_set
})

# 3. whole data preprocessing: tokenization
tokenizer = AutoTokenizer.from_pretrained("./llama-3.2-1B/")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True

print(type(tokenizer))
def process_function(example):
    prompt = f"This is an NLI task. I'll give you two sentences which are the premise and the hypothesis respectively.\n \
              If the hypothesis can be inferred from the premise, the answer is entailment.\n \
              If the hypothesis is inconsistent with the premise, the answer is contradiction.\n \
              If the hypothesis is unrelated to premise, the answer is neutral.\n\n \
              ---\n \
              Important Note: You can only response ONE word among entailment, neutral and contradiction.\n \
              ---\n\n \
              The premise: { example['premise'] }\n \
              The hypothesis: { example['hypothesis'] }\n \
              The answer: "
    
    tokenized_example = tokenizer(text=prompt, max_length=256, padding="max_length")
    tokenized_example["labels"] = example["label"]
    return tokenized_example

tokenized_datasets = datasets.map(process_function, batched=False, remove_columns=train_set.column_names)

# 4. create model
model = LlamaForSequenceClassification.from_pretrained("./llama-3.2-1B/", num_labels=3, quantization_config=bnb_config) # bnb
model.config.pad_token_id = tokenizer.pad_token_id

# 4.1 Lora
model = prepare_model_for_kbit_training(model)  # load bnb
model = get_peft_model(model, lora_config)  # load lora

# 5. define evaluation metrics
acc_metric = evaluate.load("./metric_accuracy.py")

def eval_metric(eval_predict):
    predictions, labels = eval_predict
    predictions = predictions.argmax(axis=-1)
    acc = acc_metric.compute(predictions=predictions, references=labels)
    return acc

# 6. define training arguments
train_args = TrainingArguments(output_dir="./checkpoints",
                               per_device_train_batch_size=16,
                               gradient_checkpointing=False,
                               per_device_eval_batch_size=32,
                               num_train_epochs=1,
                               logging_steps=100,
                               eval_strategy="steps",
                               eval_steps=1000,
                               save_strategy="steps",
                               save_steps=1000,
                               save_total_limit=3,
                               learning_rate=2e-5,
                               weight_decay=0.001,
                               metric_for_best_model="accuracy",
                               load_best_model_at_end=True,
                               report_to="tensorboard",
                               fp16=False,
                               optim="paged_adamw_8bit"
                               )

# 7. create trainer
trainer = Trainer(model=model, 
                  args=train_args, 
                  tokenizer=tokenizer,
                  train_dataset=tokenized_datasets["train"], 
                  eval_dataset=tokenized_datasets["validation"], 
                  data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                  compute_metrics=eval_metric)

# 8. train
trainer.train()