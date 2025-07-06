#!/usr/bin/env python3
"""
Inference Script for Finetuned Llama3 Medical Assistant
"""

import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
import json
from typing import List, Dict, Any

class MedicalAssistantInference:
    """
    Inference class for the finetuned medical assistant
    """
    
    def __init__(self, 
                 model_path: str = "medical_llama3_finetuned",
                 max_seq_length: int = 2048,
                 dtype: str = None,
                 load_in_4bit: bool = True):
        
        self.model_path = model_path
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the finetuned model"""
        print(f"Loading finetuned model from: {self.model_path}")
        
        # Load model and tokenizer
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_path,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
        )
        
        print("Model loaded successfully!")
        return self.model, self.tokenizer
    
    def create_prompt(self, patient_query: str, context: str = "") -> str:
        """Create a prompt for the medical assistant"""
        instruction = "You are a helpful medical assistant. Please respond to the patient's query in a professional and empathetic manner."
        
        if context:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\nPatient: {context}\nPatient: {patient_query}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\nPatient: {patient_query}\n\n### Response:\n"
        
        return prompt
    
    def generate_response(self, 
                         prompt: str, 
                         max_new_tokens: int = 512,
                         temperature: float = 0.7,
                         top_p: float = 0.9,
                         do_sample: bool = True) -> str:
        """Generate response from the model"""
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after the prompt)
        response = response[len(prompt):].strip()
        
        return response
    
    def chat(self, 
             patient_query: str, 
             context: str = "",
             max_new_tokens: int = 512,
             temperature: float = 0.7) -> Dict[str, str]:
        """Interactive chat with the medical assistant"""
        
        # Create prompt
        prompt = self.create_prompt(patient_query, context)
        
        # Generate response
        response = self.generate_response(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
        return {
            "patient_query": patient_query,
            "context": context,
            "assistant_response": response,
            "prompt": prompt
        }
    
    def batch_inference(self, 
                       queries: List[str], 
                       contexts: List[str] = None,
                       max_new_tokens: int = 512,
                       temperature: float = 0.7) -> List[Dict[str, str]]:
        """Run batch inference on multiple queries"""
        
        if contexts is None:
            contexts = [""] * len(queries)
        
        results = []
        
        for query, context in zip(queries, contexts):
            result = self.chat(
                patient_query=query,
                context=context,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            results.append(result)
        
        return results
    
    def save_responses(self, responses: List[Dict[str, str]], output_path: str = "inference_results.json"):
        """Save inference results to file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(responses, f, ensure_ascii=False, indent=2)
        print(f"Inference results saved to {output_path}")

def interactive_chat():
    """Interactive chat interface"""
    print("=== Medical Assistant Chat Interface ===")
    print("Type 'quit' to exit")
    print("Type 'context: <context>' to set context for next query")
    print("-" * 50)
    
    # Initialize inference
    inference = MedicalAssistantInference()
    inference.load_model()
    
    context = ""
    
    while True:
        try:
            user_input = input("\nPatient: ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            
            if user_input.lower().startswith('context:'):
                context = user_input[8:].strip()
                print(f"Context set: {context}")
                continue
            
            if not user_input:
                continue
            
            # Generate response
            result = inference.chat(user_input, context)
            
            print(f"\nAssistant: {result['assistant_response']}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def test_examples():
    """Test the model with example medical queries"""
    print("=== Testing Medical Assistant with Example Queries ===")
    
    # Initialize inference
    inference = MedicalAssistantInference()
    inference.load_model()
    
    # Example medical queries
    test_queries = [
        "I have been experiencing headaches for the past week. What could be causing this?",
        "My blood pressure has been high lately. What should I do?",
        "I have a fever and sore throat. Should I be concerned?",
        "What are the symptoms of diabetes?",
        "I'm feeling very tired and have no energy. What could this mean?"
    ]
    
    test_contexts = [
        "Patient is a 35-year-old office worker who spends long hours at computer",
        "Patient has family history of hypertension",
        "Patient is a 28-year-old with no chronic conditions",
        "",
        "Patient has been under stress at work recently"
    ]
    
    # Run batch inference
    results = inference.batch_inference(test_queries, test_contexts)
    
    # Display results
    for i, result in enumerate(results):
        print(f"\n--- Example {i+1} ---")
        print(f"Context: {result['context']}")
        print(f"Patient: {result['patient_query']}")
        print(f"Assistant: {result['assistant_response']}")
        print("-" * 50)
    
    # Save results
    inference.save_responses(results)
    
    print("\nTesting completed! Results saved to inference_results.json")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Medical Assistant Inference")
    parser.add_argument("--mode", choices=["interactive", "test"], default="test",
                       help="Mode: interactive chat or test examples")
    parser.add_argument("--model_path", default="medical_llama3_finetuned",
                       help="Path to the finetuned model")
    
    args = parser.parse_args()
    
    if args.mode == "interactive":
        interactive_chat()
    else:
        test_examples()

if __name__ == "__main__":
    main() 