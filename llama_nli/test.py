from transformers import LlamaForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, Trainer, DataCollatorWithPadding
from datasets import load_dataset, DatasetDict
from collections import Counter
from peft import PeftModel
import torch
from copy import deepcopy

tokenizer = AutoTokenizer.from_pretrained("./llama-3.2-1B/")
tokenizer.pad_token = tokenizer.eos_token

model = LlamaForSequenceClassification.from_pretrained("./llama-3.2-1B/", num_labels=3) # bnb
model.config.pad_token_id = tokenizer.pad_token_id

p_model = PeftModel.from_pretrained(model, model_id="./checkpoints/checkpoint-24471")
merge_model = p_model.merge_and_unload()

# Test data preprocessing
str2int = {'contradiction': 2, 'entailment': 0, 'neutral': 1}
int2str = {2: 'contradiction', 0: 'entailment', 1: 'neutral'}
test_set_matched = load_dataset("json", data_files="./test_multi_nli/dev_matched_sampled-1.jsonl").select_columns(["sentence1", "sentence2", "gold_label", "annotator_labels"])["train"]  # get ["train"] since select_columns returns a DatasetDict defautly
test_set_mismatched = load_dataset("json", data_files="./test_multi_nli/dev_mismatched_sampled-1.jsonl").select_columns(["sentence1", "sentence2", "gold_label", "annotator_labels"])["train"]

def test_preprocessing(example):
    example["premise"] = example["sentence1"]
    example["hypothesis"] = example["sentence2"]

    # example['annotator_labels']: list of strings / example['gold_label']: string
    if example["gold_label"] == "-":    # tie, we consider either prediction as correct
        # get the most common annotator_labels
        counter = Counter(example['annotator_labels'])
        max_count = max(counter.values())
        most_common_elements = [key for key, count in counter.items() if count == max_count]
        # set gt as both 2 labels
        example["label"] = [str2int[el] for el in most_common_elements]
    else:
        example["label"] = [str2int[example["gold_label"]]]
        
    return example

test_set_matched = test_set_matched.map(function=test_preprocessing, batched=False, remove_columns=["gold_label", "annotator_labels", "sentence1", "sentence2"])
test_set_mismatched = test_set_mismatched.map(function=test_preprocessing, batched=False, remove_columns=["gold_label", "annotator_labels", "sentence1", "sentence2"])



#---------------------------------------
def dc_dataset_for_labels(example):
    example["label"] = example["label"][0]
    return example
dc_test_set_matched = test_set_matched.map(function=dc_dataset_for_labels, batched=False)
dc_test_set_mismatched = test_set_mismatched.map(function=dc_dataset_for_labels, batched=False)

datasets = DatasetDict({
    "test_matched": dc_test_set_matched,
    "test_mismatched": dc_test_set_mismatched,
})
#----------------------------------------
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

tokenized_datasets = datasets.map(process_function, batched=False, remove_columns=test_set_matched.column_names)

# use trainer as a predictor
trainer = Trainer(model=merge_model,
                  tokenizer=tokenizer,
                  data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
                  )

test_matched_dataset_dc = tokenized_datasets["test_matched"]    # dc_test_set_matched
test_mismatched_dataset_dc = tokenized_datasets["test_mismatched"]    # dc_test_set_mismatched
predictions = trainer.predict(test_mismatched_dataset_dc)   #1

predicted_labelsint = [prediction.argmax(axis=-1) for prediction in predictions.predictions]


acc = 0
for pred_label, true_labels in zip(predicted_labelsint, test_set_mismatched['label']):  #2
    if pred_label in true_labels:
        acc += 1
acc = float(acc)/len(predicted_labelsint)
print(acc)
