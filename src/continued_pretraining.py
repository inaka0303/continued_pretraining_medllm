from transformers import AutoTokenizer, DataCollatorForLanguageModeling, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset, concatenate_datasets

model_name = "rinna/deepseek-r1-distill-qwen2.5-bakeneko-32b"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# TXTファイルの読み込み
dataset_txt = load_dataset('text', data_files={'train': 'path/to/data.txt'})
# CSVファイルの読み込み（例として"text"カラムがあると仮定）
dataset_csv = load_dataset('csv', data_files={'train': 'path/to/data.csv'})

# JSONファイルの読み込み（例として"content"というキーにテキストがあると仮定）
dataset_json = load_dataset('json', data_files={'train': 'path/to/data.json'})

# CSVやJSONのデータからテキストを統一的に抽出するための関数
def extract_text(example):
    # CSVの場合は"text"キー、JSONの場合は"content"キーなど、適宜調整してください
    if "text" in example:
        return {"text": example["text"]}
    elif "content" in example:
        return {"text": example["content"]}
    else:
        return {"text": ""}

dataset_csv = dataset_csv.map(extract_text)
dataset_json = dataset_json.map(extract_text)

# 各データセットは 'train' キーに格納されているので、これらを結合
combined_dataset = concatenate_datasets([
    dataset_txt["train"],
    dataset_csv["train"],
    dataset_json["train"]
])

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=1024)

tokenized_datasets = combined_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

model = AutoModelForCausalLM.from_pretrained(model_name)


training_args = TrainingArguments(
    output_dir="./med_continued_pretrain",
    overwrite_output_dir=True,
    num_train_epochs=3,  # エポック数はデータ量や実験結果に応じて調整
    per_device_train_batch_size=2,  # 使用するGPUメモリに合わせて設定
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=5e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
)

trainer.train()

trainer.save_model("./med_continued_pretrain_final")

