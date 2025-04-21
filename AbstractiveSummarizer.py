import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset, load_dataset
import os
import numpy as np
import json

class AbstractiveSummarizer:
  def __init__(self, model_name="facebook/bart-large-cnn", device=None, model_dir="fine_tuned_model"):
    """Initialize the abstractive summarizer"""
    self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    self.model_name = model_name
    self.model_dir = model_dir
    
    # Check if fine-tuned model exists
    if os.path.exists(model_dir):
      print(f"Loading fine-tuned model from {model_dir}")
      self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
      self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(self.device)
    else:
      print(f"Loading pre-trained model {model_name}")
      self.tokenizer = AutoTokenizer.from_pretrained(model_name)
      self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

  def summarize(self, text, max_length=500, min_length=50, length_penalty=2, num_beams=4):
    """Generate abstractive summary"""
    # Limit input to 3048 tokens to avoid model constraints
    text = text[:3048]
    inputs = self.tokenizer(text, return_tensors="pt", max_length=len(text), truncation=True)
    inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}

    summary_ids = self.model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        max_length=max_length,
        min_length=min_length,
        length_penalty=length_penalty,
        num_beams=num_beams,
        early_stopping=True,
        no_repeat_ngram_size=3,
        decoder_start_token_id=self.model.config.decoder_start_token_id
    )

    summary = self.tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    return summary
    
  def preprocess_data(self, examples):
    """Preprocess data for training"""
    inputs = self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=1024)
    outputs = self.tokenizer(examples["summary"], padding="max_length", truncation=True, max_length=128)
    
    batch = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "labels": outputs.input_ids,
    }
    
    # Replace padding token id with -100 so it's ignored in loss computation
    batch["labels"] = [
        [-100 if token == self.tokenizer.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]
    
    return batch
    
  def compute_metrics(self, eval_preds):
    """Compute ROUGE metrics"""
    from rouge import Rouge
    rouge = Rouge()
    
    preds, labels = eval_preds
    
    # Replace -100 with pad token id
    labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
    
    # Decode predictions and labels
    decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge scores
    scores = rouge.get_scores(decoded_preds, decoded_labels, avg=True)
    
    return {
        "rouge-1": scores["rouge-1"]["f"],
        "rouge-2": scores["rouge-2"]["f"],
        "rouge-l": scores["rouge-l"]["f"],
    }
  
  def fine_tune(self, train_data, validation_data=None, output_dir="fine_tuned_model", 
                epochs=7, batch_size=4, learning_rate=5e-5):
    """
    Fine-tune the model on custom data
    
    Args:
      train_data: List of dicts with 'text' and 'summary' keys
      validation_data: Optional list of dicts with 'text' and 'summary' keys
      output_dir: Directory to save the fine-tuned model
      epochs: Number of training epochs
      batch_size: Training batch size
      learning_rate: Learning rate for training
    """
    # Convert data to datasets
    train_dataset = Dataset.from_list(train_data)
    
    if validation_data:
      val_dataset = Dataset.from_list(validation_data)
    else:
      # Use a small portion of training data for validation if not provided
      split_dataset = train_dataset.train_test_split(test_size=0.1)
      train_dataset = split_dataset["train"]
      val_dataset = split_dataset["test"]
    
    # Preprocess datasets
    tokenized_train = train_dataset.map(
        lambda x: self.preprocess_data(x),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    tokenized_val = val_dataset.map(
        lambda x: self.preprocess_data(x),
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    # Set up training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=epochs,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=self.model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=self.tokenizer,
        compute_metrics=self.compute_metrics,
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    self.model.save_pretrained(output_dir)
    self.tokenizer.save_pretrained(output_dir)
    self.model_dir = output_dir
    
    print(f"Model fine-tuned and saved to {output_dir}")
    return trainer
    
  def check_and_fine_tune(self, dataset_path="training_data.json", model_dir="fine_tuned_model", 
                          epochs=3, batch_size=4):
    """
    Check if fine-tuned model exists, if not, fine-tune using the provided dataset
    
    Args:
      dataset_path: Path to dataset file (JSON or CSV)
      model_dir: Directory to save the fine-tuned model
      epochs: Number of training epochs
      batch_size: Training batch size
    """
    # If model already exists, nothing to do
    if os.path.exists(model_dir):
      print(f"Fine-tuned model already exists at {model_dir}")
      return
    
    print(f"Fine-tuned model not found. Loading dataset from {dataset_path}")
    
    # Load dataset based on file extension
    if dataset_path.endswith('.json'):
      try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
          data = json.load(f)
      except Exception as e:
        print(f"Error loading JSON dataset: {e}")
        return
    elif dataset_path.endswith('.csv'):
      try:
        dataset = load_dataset('csv', data_files=dataset_path)
        data = [{"text": row["text"], "summary": row["summary"]} for row in dataset["train"]]
      except Exception as e:
        print(f"Error loading CSV dataset: {e}")
        return
    else:
      print(f"Unsupported dataset format: {dataset_path}")
      return
    
    # Verify dataset format
    if not all('text' in example and 'summary' in example for example in data):
      print("Dataset must contain 'text' and 'summary' fields")
      return
    
    # Fine-tune the model
    print(f"Fine-tuning model on {len(data)} examples...")
    self.fine_tune(
      train_data=data,
      output_dir=model_dir, 
      epochs=epochs,
      batch_size=batch_size
    )
