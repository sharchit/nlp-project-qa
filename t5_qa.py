import json
import torch
from transformers import (
    T5ForConditionalGeneration,  
    T5Tokenizer, 
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from sklearn.model_selection import train_test_split
from datasets import Dataset
from rouge import Rouge
from bert_score import score as bert_score
from torch.utils.data import DataLoader
from csv import DictReader

def prepare_dataset(data, test_size=0.2):
    contexts = [item['context'] for item in data]
    questions = [item['ques'] for item in data]
    answers = [item['ans'] for item in data]
    
    splits = train_test_split(contexts, questions, answers, 
                            test_size=test_size, random_state=42)
    return {
        'train': {'context': splits[0], 'question': splits[2], 'answer': splits[4]},
        'test': {'context': splits[1], 'question': splits[3], 'answer': splits[5]}
    }

def tokenize_function(tokenizer, examples):
    inputs = [f"generate answer: context: {c} question: {q}" for c, q in zip(examples['context'], examples['question'])]  # T5-style prefix
    targets = examples['answer']
    
    model_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding='max_length',
        add_special_tokens=True  # Explicitly add special tokens
    )
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=128,
            truncation=True,
            padding='max_length',
            add_special_tokens=True
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def generate_predictions(model, tokenizer, dataset, device="cuda"):
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    eval_loader = DataLoader(
        dataset,
        batch_size=8,
        collate_fn=data_collator
    )
    
    model.to(device)
    model.eval()
    
    predictions = []
    references = []
    
    with torch.no_grad():
        for batch in eval_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            generated_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=128,
                num_beams=4,
                early_stopping=True  # Added for T5 optimization
            )
            
            batch_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            batch_refs = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

            for idx, pred in enumerate(batch_preds):
                if pred == "":
                    batch_preds[idx] = "empty"
            
            predictions.extend(batch_preds)
            references.extend(batch_refs)
            
    print(len(predictions), len(references), predictions[0], references[0])
    return predictions, references

def compute_metrics(predictions, references):
    results = {}
    
    rouge = Rouge()
    rouge_scores = rouge.get_scores(predictions, references, avg=True)
    results['rouge-1'] = rouge_scores['rouge-1']['f']
    results['rouge-2'] = rouge_scores['rouge-2']['f']
    results['rouge-l'] = rouge_scores['rouge-l']['f']

    P, R, F1 = bert_score(predictions, references, lang='en')
    results['bertscore_precision'] = P.mean().item()
    results['bertscore_recall'] = R.mean().item()
    results['bertscore_f1'] = F1.mean().item()
    
    return results, predictions

def run_t5_pipeline(model_id, data, output_file='t5_results.json'):
    dataset = prepare_dataset(data)
    results = {}
    
    tokenizer = T5Tokenizer.from_pretrained(model_id) 
    model = T5ForConditionalGeneration.from_pretrained(model_id) 
    
    train_dataset = Dataset.from_dict(dataset['train']).map(
        lambda x: tokenize_function(tokenizer, x),
        batched=True,
        remove_columns=['context', 'question', 'answer']
    )
    test_dataset = Dataset.from_dict(dataset['test']).map(
        lambda x: tokenize_function(tokenizer, x),
        batched=True,
        remove_columns=['context', 'question', 'answer']
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir='./t5_results',  
        evaluation_strategy='no',
        learning_rate=5e-4,  # T5 typically uses slightly higher learning rate
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=5,
        save_strategy="no",
        weight_decay=0.01,
        predict_with_generate=True,
        gradient_accumulation_steps=2  # Added for T5 stability
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        tokenizer=tokenizer
    )
    
    trainer.train()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictions, references = generate_predictions(model, tokenizer, test_dataset, device)
        
    metrics, preds = compute_metrics(predictions, references)
    
    results[model_id] = {
        'metrics': metrics,
    }

    with open("check.json", 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results

if __name__ == "__main__":
    with open("final_dataset.csv", 'r') as f:
        dict_reader = DictReader(f)
        data = list(dict_reader)
    
    results = run_t5_pipeline("t5-large", data)  
        
    with open('t5_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
