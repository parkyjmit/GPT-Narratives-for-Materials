
import argparse


def main(args):
    import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    from transformers import BertTokenizer
    from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer, AutoModelForCausalLM
    from datasets import load_dataset

    import numpy as np
    import evaluate
    from dataclasses import dataclass
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
    from typing import Optional, Union
    import torch
    from transformers import BitsAndBytesConfig
    from peft import LoraConfig

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    datasets = load_dataset('json', data_files={'train': args.train_path, 'validation': args.valid_path})
    #Model
    model = AutoModelForMultipleChoice.from_pretrained(args.model_name)#, quantization_config=quantization_config)

    #Tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    num_choice = len(datasets['train'][0].keys()) - 3
    ending_names = [f"ending{i}" for i in range(num_choice)]


    def preprocess_function(examples):
        first_sentences = [[context] * num_choice for context in examples["sent1"]]
        question_headers = examples["sent2"]
        second_sentences = [
            [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
        ]

        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])

        tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True, max_length=512)
        return {k: [v[i : i + num_choice] for i in range(0, len(v), num_choice)] for k, v in tokenized_examples.items()}

    tokenized_datasets = datasets.map(preprocess_function, batched=True, num_proc=24)


    @dataclass
    class DataCollatorForMultipleChoice:
        """
        Data collator that will dynamically pad the inputs for multiple choice received.
        """

        tokenizer: PreTrainedTokenizerBase
        padding: Union[bool, str, PaddingStrategy] = True
        max_length: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None

        def __call__(self, features):
            label_name = "label" if "label" in features[0].keys() else "labels"
            labels = [feature.pop(label_name) for feature in features]
            batch_size = len(features)
            num_choices = len(features[0]["input_ids"])
            flattened_features = [
                [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
            ]
            flattened_features = sum(flattened_features, [])

            batch = self.tokenizer.pad(
                flattened_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )

            batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
            batch["labels"] = torch.tensor(labels, dtype=torch.int64)
            return batch
        
    accuracy = evaluate.load('accuracy')


    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir=args.model_name.split('/')[-1][:3] + args.train_path.split('/')[-1].split('_')[0],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='/mnt/hdd1/LaMDa/comp_data_train.json')
    parser.add_argument('--valid_path', type=str, default='/mnt/hdd1/LaMDa/comp_data_valid.json')
    parser.add_argument('--model_name', type=str, default="allenai/scibert_scivocab_cased")
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    main(args)