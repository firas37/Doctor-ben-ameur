#!/usr/bin/env python3
"""
Llama3 Finetuning Script using Unsloth
Medical Dialogue Assistant Training
"""
import unsloth
import os
import torch
from datasets import load_from_disk
from trl import SFTTrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
import wandb
from datetime import datetime
import json

class Llama3MedicalFinetuner:
    """
    Finetune Llama3 model for medical dialogue assistance using Unsloth
    """
    
    def __init__(self, 
                 model_name: str = "unsloth/llama-3-8b-bnb-4bit",
                 dataset_path: str = "medical_dialogue_dataset",
                 output_dir: str = "medical_llama3_finetuned",
                 max_seq_length: int = 2048,
                 dtype: str = None,
                 load_in_4bit: bool = True):
        
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        
        # Training parameters
        self.training_args = None
        self.trainer = None
        self.model = None
        self.tokenizer = None
        
    def setup_wandb(self, project_name: str = "medical-llama3-finetuning"):
        """Setup Weights & Biases for experiment tracking"""
        wandb.init(
            project=project_name,
            name=f"medical-llama3-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config={
                "model_name": self.model_name,
                "max_seq_length": self.max_seq_length,
                "load_in_4bit": self.load_in_4bit,
                "dataset_path": self.dataset_path
            }
        )
        print(f"WandB initialized: {wandb.run.name}")
    
    def load_model_and_tokenizer(self):
        """Load the Llama3 model and tokenizer using Unsloth"""
        print(f"Loading model: {self.model_name}")
        
        # Load model and tokenizer with Unsloth optimizations
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
        )
        
        # Add LoRA adapters for efficient finetuning
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,  # Rank
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        
        print("Model and tokenizer loaded successfully!")
        return self.model, self.tokenizer
    
    def load_dataset(self):
        """Load the prepared medical dialogue dataset"""
        print(f"Loading dataset from: {self.dataset_path}")
        
        if os.path.exists(self.dataset_path):
            dataset = load_from_disk(self.dataset_path)
        else:
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
        
        print(f"Dataset loaded: {len(dataset)} examples")
        print(f"Dataset features: {dataset.features}")
        
        return dataset
    
    def create_prompt_template(self, example):
        """Create prompt template for instruction tuning"""
        # Format: Instruction + Input + Output
        if example.get('input', '').strip():
            prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
        else:
            prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
        
        return prompt
    
    def setup_training_arguments(self, 
                                num_train_epochs: int = 3,
                                per_device_train_batch_size: int = 2,
                                gradient_accumulation_steps: int = 4,
                                learning_rate: float = 2e-4,
                                warmup_steps: int = 100,
                                logging_steps: int = 10,
                                save_steps: int = 500,
                                eval_steps: int = 500,
                                save_total_limit: int = 3):
        """Setup training arguments"""
        
        self.training_args = SFTTrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            save_total_limit=save_total_limit,
            warmup_steps=warmup_steps,
            evaluation_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            dataloader_num_workers=4,
            group_by_length=True,
            ddp_find_unused_parameters=False,
        )
        
        print("Training arguments configured!")
        return self.training_args
    
    def setup_trainer(self, dataset):
        """Setup the SFT trainer"""
        print("Setting up SFT trainer...")
        
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            args=self.training_args,
            max_seq_length=self.max_seq_length,
            dataset_text_field="text",
            packing=False,
            peft_config=None,
        )
        
        print("SFT trainer setup completed!")
        return self.trainer
    
    def train(self):
        """Execute the training process"""
        print("Starting training...")
        
        # Train the model
        self.trainer.train()
        
        # Save the final model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"Training completed! Model saved to {self.output_dir}")
        
        # Log final metrics
        if wandb.run:
            wandb.finish()
    
    def evaluate_model(self, test_dataset=None):
        """Evaluate the trained model"""
        print("Evaluating model...")
        
        if test_dataset is None:
            # Use a subset of training data for evaluation
            dataset = self.load_dataset()
            test_dataset = dataset.select(range(min(100, len(dataset))))
        
        # Run evaluation
        eval_results = self.trainer.evaluate(test_dataset)
        
        print("Evaluation results:")
        for key, value in eval_results.items():
            print(f"  {key}: {value}")
        
        return eval_results
    
    def save_training_config(self):
        """Save training configuration for reproducibility"""
        config = {
            "model_name": self.model_name,
            "dataset_path": self.dataset_path,
            "output_dir": self.output_dir,
            "max_seq_length": self.max_seq_length,
            "load_in_4bit": self.load_in_4bit,
            "training_args": self.training_args.to_dict() if self.training_args else None,
            "timestamp": datetime.now().isoformat()
        }
        
        config_path = os.path.join(self.output_dir, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Training config saved to {config_path}")

def main():
    """Main function to run the finetuning process"""
    
    # Initialize the finetuner
    finetuner = Llama3MedicalFinetuner(
        model_name="unsloth/llama-3-8b-bnb-4bit",
        dataset_path="medical_dialogue_dataset",
        output_dir="medical_llama3_finetuned",
        max_seq_length=2048,
        load_in_4bit=True
    )
    
    # Setup WandB for experiment tracking
    finetuner.setup_wandb()
    
    # Load model and tokenizer
    finetuner.load_model_and_tokenizer()
    
    # Load dataset
    dataset = finetuner.load_dataset()
    
    # Setup training arguments
    finetuner.setup_training_arguments(
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        eval_steps=500
    )
    
    # Setup trainer
    finetuner.setup_trainer(dataset)
    
    # Save training configuration
    finetuner.save_training_config()
    
    # Start training
    finetuner.train()
    
    # Evaluate the model
    finetuner.evaluate_model()
    
    print("Finetuning process completed successfully!")

if __name__ == "__main__":
    main() 