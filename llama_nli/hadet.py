from transformers import LlamaForSequenceClassification, AutoTokenizer, Trainer, DataCollatorWithPadding
from datasets import load_dataset, Dataset
from peft import PeftModel

# local dataset
datasets = load_dataset("./datasets--potsawee--wiki_bio_gpt3_hallucination", trust_remote_code=True)
dataset = datasets["test"]

# flatten the dataset
dataset_dict = {}
hypothesis = []
premise = []
label = []

for i in range(len(dataset)):
    for j in range(len(dataset["gpt3_sentences"][i])):
        hypothesis.append(dataset["gpt3_sentences"][i][j])
        premise.append(dataset["wiki_bio_text"][i])
        label.append(dataset["annotation"][i][j])

dataset_dict["hypothesis"] = hypothesis
dataset_dict["premise"] = premise
dataset_dict["label"] = label

dataset = Dataset.from_dict(dataset_dict)   # flattened dataset

# tokenize the dataset
tokenizer = AutoTokenizer.from_pretrained("./llama-3.2-1B/")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True

model = LlamaForSequenceClassification.from_pretrained("./llama-3.2-1B/", num_labels=3) # bnb
model.config.pad_token_id = tokenizer.pad_token_id
p_model = PeftModel.from_pretrained(model, model_id="./checkpoints/checkpoint-24471")
merge_model = p_model.merge_and_unload()

str2int = {'major_inaccurate': 2, 'accurate': 0, 'minor_inaccurate': 1}

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
    tokenized_example["labels"] = str2int[example["label"]]
    return tokenized_example

tokenized_dataset = dataset.map(process_function, batched=False, remove_columns=dataset.column_names)

# use trainer as a predictor
trainer = Trainer(model=merge_model,
                  tokenizer=tokenizer,
                  data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
                  )

predictions = trainer.predict(tokenized_dataset)

predicted_labelsint = [prediction.argmax(axis=-1) for prediction in predictions.predictions]

# metrics
tp = 0
tn = 0
fp = 0
fn = 0  # p:0, n:1/2

for pred_label, true_label in zip(predicted_labelsint, tokenized_dataset['labels']):
    if pred_label in [1, 2] and true_label in [1, 2]:   # non-factual for (0.5 and 1)
        tn = tn + 1
    elif pred_label == true_label:  # both 0, factual
        tp = tp + 1
    elif true_label == 0:
        fn = fn + 1
    else:   # true_label != 0
        fp = fp + 1

precision = float(tp)/(tp+fp)
recall = float(tp)/(tp+fn)
f1 = (2 * precision * recall) / (precision + recall)
acc = float(tp+tn)/len(predicted_labelsint)

print(f"tn={tn}, tp={tp}, fn={fn}, fp={fp}")
print(f"acc: {acc}")
print(f"precision: {precision}")
print(f"recall: {recall}")
print(f"f1: {f1}")