import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from transformers.optimization import get_cosine_schedule_with_warmup
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Model configuration class"""
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    max_length: int = 2048
    learning_rate: float = 5e-5
    batch_size: int = 8
    num_epochs: int = 15
    warmup_ratio: float = 0.15
    weight_decay: float = 0.1
    dropout: float = 0.1
    random_seed: int = 2024
    beam_size: int = 4
    length_penalty: float = 3.0
    no_repeat_ngram_size: int = 3
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    max_new_tokens: int = 512  # Default value, adjusted based on dataset

class PlanOutputModel:
    """Plan-Output model: jointly generates plan and summary"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Add pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Set random seed
        torch.manual_seed(config.random_seed)
        
    def prepare_training_data(self, data_path: str) -> List[Dict]:
        """Prepare training data"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        formatted_data = []
        for item in data:
            document = item['document']
            plan_questions = item.get('plan_questions', [])
            summary = item['summary']
            
            # Plan-Output model: input contains only document, output is plan+summary sequence [b;s]
            input_text = f"Generate a lay summary for the following document.\nDocument: {document}\nLay Summary:"
            
            # Format output (plan + summary) - [b;s] format from paper
            plan_text = " ".join([f"Question: {q}" for q in plan_questions])
            output_text = f"{plan_text} {summary}"
            
            formatted_data.append({
                'input_text': input_text,
                'output_text': output_text,
                'input_ids': None,
                'labels': None
            })
            
        return formatted_data
    
    def tokenize_data(self, formatted_data: List[Dict]) -> List[Dict]:
        """Tokenize data"""
        for item in formatted_data:
            # Tokenize input and output
            full_text = item['input_text'] + item['output_text']
            encoded = self.tokenizer(
                full_text,
                truncation=True,
                padding=False,
                max_length=self.config.max_length,
                return_tensors=None
            )
            
            # Calculate input length for generating labels
            input_encoded = self.tokenizer(
                item['input_text'],
                truncation=True,
                padding=False,
                max_length=self.config.max_length,
                return_tensors=None
            )
            input_length = len(input_encoded['input_ids'])
            
            # Create labels (input part is -100, output part is token ids)
            labels = [-100] * input_length + encoded['input_ids'][input_length:]
            
            item['input_ids'] = encoded['input_ids']
            item['labels'] = labels
            
        return formatted_data
    
    def train(self, train_data_path: str, val_data_path: str, output_dir: str):
        """Train model"""
        logger.info("Preparing training data...")
        train_data = self.prepare_training_data(train_data_path)
        train_data = self.tokenize_data(train_data)
        
        val_data = self.prepare_training_data(val_data_path)
        val_data = self.tokenize_data(val_data)
        
        # Create datasets
        train_dataset = PlanDataset(train_data)
        val_dataset = PlanDataset(val_data)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            seed=self.config.random_seed,
            remove_unused_columns=False,
            dataloader_drop_last=True,
            fp16=True,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Create Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            processing_class=self.tokenizer,
        )
        
        logger.info("Starting training...")
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Model saved to {output_dir}")
    
    def generate_summary(self, document: str, plan_questions: List[str] = None) -> str:
        """Generate summary"""
        # Plan-Output model: input contains only document, generates plan+summary sequence
        input_text = f"Generate a lay summary for the following document.\nDocument: {document}\nLay Summary:"
        
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                num_beams=self.config.beam_size,
                length_penalty=self.config.length_penalty,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract generated part (remove input part)
        generated_content = generated_text[len(input_text):].strip()
        
        # Parse generated [b;s] sequence, extract summary part
        # Improved separation logic: find continuous text paragraphs, questions usually end with "?", summary is paragraph form
        if "Question:" in generated_content:
            # Split all Question paragraphs
            parts = generated_content.split("Question:")
            
            # Skip first empty part, process all Question parts
            questions = []
            summary_candidates = []
            
            for i, part in enumerate(parts[1:], 1):
                part = part.strip()
                if part:
                    # If ends with question mark, likely a question
                    if part.endswith('?'):
                        questions.append(part)
                    else:
                        # Check if contains question mark, if yes, separate question and following content
                        if '?' in part:
                            question_end = part.rfind('?') + 1
                            question_part = part[:question_end].strip()
                            remaining_part = part[question_end:].strip()
                            
                            if question_part:
                                questions.append(question_part)
                            if remaining_part:
                                summary_candidates.append(remaining_part)
                        else:
                            # No question mark, possibly summary content
                            summary_candidates.append(part)
            
            # Choose longest candidate as summary, or combine all non-question content
            if summary_candidates:
                # Find longest continuous text segment as summary
                summary = max(summary_candidates, key=len).strip()
            else:
                # If no clear summary candidates, return all generated content
                summary = generated_content
        else:
            # If no Question markers, entire generated content is summary
            summary = generated_content
        
        return summary


class PlanInputModel:
    """Plan-Input model: separately trains PG and SG modules"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # PG module - plan generation
        self.pg_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # SG module - summary generation  
        self.sg_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Add pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Set random seed
        torch.manual_seed(config.random_seed)
    
    def prepare_pg_data(self, data_path: str) -> List[Dict]:
        """Prepare training data for PG module"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        formatted_data = []
        for item in data:
            document = item['document']
            plan_questions = item.get('plan_questions', [])
            
            input_text = f"Generate planning questions for the following document.\nDocument: {document}\nPlanning Questions:"
            output_text = " ".join([f"Question: {q}" for q in plan_questions])
            
            formatted_data.append({
                'input_text': input_text,
                'output_text': output_text,
                'input_ids': None,
                'labels': None
            })
            
        return formatted_data
    
    def prepare_sg_data(self, data_path: str) -> List[Dict]:
        """Prepare training data for SG module"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        formatted_data = []
        for item in data:
            document = item['document']
            plan_questions = item.get('plan_questions', [])
            summary = item['summary']
            
            plan_text = " ".join([f"Question: {q}" for q in plan_questions])
            input_text = f"Generate a lay summary for the following document based on the plan questions.\nDocument: {document}\nPlanning Questions: {plan_text}\nEnsure that the generated summary sequentially answers the plan questions.\nLay Summary:"
            output_text = summary
            
            formatted_data.append({
                'input_text': input_text,
                'output_text': output_text,
                'input_ids': None,
                'labels': None
            })
            
        return formatted_data
    
    def tokenize_data(self, formatted_data: List[Dict]) -> List[Dict]:
        """Tokenize data"""
        for item in formatted_data:
            full_text = item['input_text'] + item['output_text']
            encoded = self.tokenizer(
                full_text,
                truncation=True,
                padding=False,
                max_length=self.config.max_length,
                return_tensors=None
            )
            
            input_encoded = self.tokenizer(
                item['input_text'],
                truncation=True,
                padding=False,
                max_length=self.config.max_length,
                return_tensors=None
            )
            input_length = len(input_encoded['input_ids'])
            
            labels = [-100] * input_length + encoded['input_ids'][input_length:]
            
            item['input_ids'] = encoded['input_ids']
            item['labels'] = labels
            
        return formatted_data
    
    def train_pg(self, train_data_path: str, val_data_path: str, output_dir: str):
        """Train PG module"""
        logger.info("Training PG module...")
        
        train_data = self.prepare_pg_data(train_data_path)
        train_data = self.tokenize_data(train_data)
        
        val_data = self.prepare_pg_data(val_data_path)
        val_data = self.tokenize_data(val_data)
        
        train_dataset = PlanDataset(train_data)
        val_dataset = PlanDataset(val_data)
        
        training_args = TrainingArguments(
            output_dir=os.path.join(output_dir, "pg"),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            seed=self.config.random_seed,
            remove_unused_columns=False,
            dataloader_drop_last=True,
            fp16=True,
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        trainer = Trainer(
            model=self.pg_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            processing_class=self.tokenizer,
        )
        
        trainer.train()
        trainer.save_model()
        
        logger.info("PG module training completed")
    
    def train_sg(self, train_data_path: str, val_data_path: str, output_dir: str):
        """Train SG module"""
        logger.info("Training SG module...")
        
        train_data = self.prepare_sg_data(train_data_path)
        train_data = self.tokenize_data(train_data)
        
        val_data = self.prepare_sg_data(val_data_path)
        val_data = self.tokenize_data(val_data)
        
        train_dataset = PlanDataset(train_data)
        val_dataset = PlanDataset(val_data)
        
        training_args = TrainingArguments(
            output_dir=os.path.join(output_dir, "sg"),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            seed=self.config.random_seed,
            remove_unused_columns=False,
            dataloader_drop_last=True,
            fp16=True,
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        trainer = Trainer(
            model=self.sg_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            processing_class=self.tokenizer,
        )
        
        trainer.train()
        trainer.save_model()
        
        logger.info("SG module training completed")
    
    def generate_plan(self, document: str) -> List[str]:
        """Generate plan using PG module"""
        input_text = f"Generate planning questions for the following document.\nDocument: {document}\nPlanning Questions:"
        
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length
        ).to(self.pg_model.device)
        
        with torch.no_grad():
            outputs = self.pg_model.generate(
                **inputs,
                max_new_tokens=min(256, self.config.max_new_tokens // 2),  # Plan generation uses fewer tokens
                num_beams=self.config.beam_size,
                length_penalty=self.config.length_penalty,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        plan_text = generated_text[len(input_text):].strip()
        
        # Parse generated questions
        questions = []
        for line in plan_text.split("Question:"):
            line = line.strip()
            if line and line != "":
                questions.append(line)
                
        return questions
    
    def generate_summary(self, document: str, plan_questions: List[str] = None) -> str:
        """Generate summary using SG module"""
        if plan_questions is None:
            # First use PG module to generate plan
            plan_questions = self.generate_plan(document)
        
        plan_text = " ".join([f"Question: {q}" for q in plan_questions])
        input_text = f"Generate a lay summary for the following document based on the plan questions.\nDocument: {document}\nPlanning Questions: {plan_text}\nEnsure that the generated summary sequentially answers the plan questions.\nLay Summary:"
        
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length
        ).to(self.sg_model.device)
        
        with torch.no_grad():
            outputs = self.sg_model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                num_beams=self.config.beam_size,
                length_penalty=self.config.length_penalty,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        summary = generated_text[len(input_text):].strip()
        
        return summary
    
    def load_trained_models(self, pg_model_path: str, sg_model_path: str):
        """Load trained PG and SG modules"""
        logger.info(f"Loading PG module: {pg_model_path}")
        self.pg_model = AutoModelForCausalLM.from_pretrained(
            pg_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        logger.info(f"Loading SG module: {sg_model_path}")
        self.sg_model = AutoModelForCausalLM.from_pretrained(
            sg_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )


class PlanDataset(torch.utils.data.Dataset):
    """Custom dataset class"""
    
    def __init__(self, data: List[Dict]):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'labels': torch.tensor(item['labels'], dtype=torch.long)
        } 