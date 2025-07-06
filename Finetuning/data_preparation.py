import pandas as pd
import json
from datasets import Dataset
import yaml
from typing import List, Dict, Any
import os

class MedicalDialogueDataPreparator:
    """
    Prepare medical dialogue data for Llama3 finetuning with Unsloth
    """
    
    def __init__(self, data_path: str = "../datacreation/dialogues.csv"):
        self.data_path = data_path
        self.formatted_data = []
        
    def load_data(self) -> pd.DataFrame:
        """Load the medical dialogue dataset with error handling"""
        print(f"Loading data from {self.data_path}")
        
        # First, let's check if the file exists
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"File not found: {self.data_path}")
        
        try:
            # Method 1: Try with TAB separator (TSV file)
            df = pd.read_csv(
                self.data_path,
                sep='\t',  # Use tab separator
                encoding='utf-8',
                quoting=1,  # QUOTE_ALL
                escapechar='\\',
                on_bad_lines='skip'  # Skip bad lines (new parameter)
            )
            print("Successfully loaded with tab separator")
        except Exception as e1:
            print(f"Method 1 (tab separator) failed: {e1}")
            try:
                # Method 2: Try with comma separator
                df = pd.read_csv(
                    self.data_path,
                    sep=',',
                    encoding='utf-8',
                    quotechar='"',
                    on_bad_lines='skip'
                )
                print("Successfully loaded with comma separator")
            except Exception as e2:
                print(f"Method 2 (comma separator) failed: {e2}")
                try:
                    # Method 3: Try with python engine and tab separator
                    df = pd.read_csv(
                        self.data_path,
                        sep='\t',
                        engine='python',
                        encoding='utf-8',
                        on_bad_lines='skip'
                    )
                    print("Successfully loaded with python engine and tab separator")
                except Exception as e3:
                    print(f"Method 3 failed: {e3}")
                    # Method 4: Try to inspect the file first
                    print("Trying to inspect the file structure...")
                    self.inspect_csv_file()
                    raise Exception("All methods failed to read the CSV file")
        
        print(f"Loaded {len(df)} dialogues")
        print(f"Columns found: {df.columns.tolist()}")
        print(f"Shape: {df.shape}")
        print(f"First few rows:")
        print(df.head(2))
        return df
    
    def inspect_csv_file(self):
        """Inspect the CSV file to understand its structure"""
        print("Inspecting CSV file structure...")
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:10]  # Read first 10 lines
                for i, line in enumerate(lines):
                    print(f"Line {i+1}: {repr(line)}")
                    print(f"  Fields count: {len(line.split(','))}")
        except Exception as e:
            print(f"Error inspecting file: {e}")
    
    def clean_csv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the CSV data"""
        print("Cleaning data...")
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Print column info
        print(f"Columns: {df.columns.tolist()}")
        print(f"Shape: {df.shape}")
        
        # If there are unexpected columns, we might need to map them
        expected_columns = ['Description', 'Patient', 'Doctor']
        
        # Check if we have the expected columns
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            print(f"Missing expected columns: {missing_cols}")
            print("Available columns:", df.columns.tolist())
            
            # Try to map columns if they have similar names
            column_mapping = {}
            for expected_col in expected_columns:
                for actual_col in df.columns:
                    if expected_col.lower() in actual_col.lower() or actual_col.lower() in expected_col.lower():
                        column_mapping[actual_col] = expected_col
                        break
            
            if column_mapping:
                print(f"Applying column mapping: {column_mapping}")
                df = df.rename(columns=column_mapping)
        
        return df
    
    def format_for_chat(self, row: pd.Series) -> Dict[str, str]:
        """
        Format a dialogue row into chat format for instruction tuning
        """
        # Create a conversation format
        conversation = []
        
        # Add patient's description if available
        if 'Description' in row and pd.notna(row.get('Description', '')) and str(row['Description']).strip():
            conversation.append(f"Patient: {str(row['Description']).strip()}")
        
        # Add patient's message
        if 'Patient' in row and pd.notna(row.get('Patient', '')) and str(row['Patient']).strip():
            conversation.append(f"Patient: {str(row['Patient']).strip()}")
        
        # Get doctor's response
        doctor_response = ""
        if 'Doctor' in row and pd.notna(row.get('Doctor', '')):
            doctor_response = str(row['Doctor']).strip()
        
        # Join conversation
        conversation_text = "\n".join(conversation)
        
        # Create instruction format
        instruction = "You are a helpful medical assistant. Please respond to the patient's query in a professional and empathetic manner."
        
        return {
            "instruction": instruction,
            "input": conversation_text,
            "output": doctor_response,
            "conversation": conversation_text
        }
    
    def create_training_data(self, max_samples: int = None) -> List[Dict[str, str]]:
        """Create training data in the required format"""
        df = self.load_data()
        df = self.clean_csv_data(df)
        
        if max_samples:
            df = df.head(max_samples)
        
        formatted_data = []
        
        for idx, row in df.iterrows():
            try:
                formatted_row = self.format_for_chat(row)
                if formatted_row['output']:  # Only include if there's a doctor response
                    formatted_data.append(formatted_row)
            except Exception as e:
                print(f"Error formatting row {idx}: {e}")
                print(f"Row data: {row}")
                continue
        
        print(f"Created {len(formatted_data)} training examples")
        return formatted_data
    
    def save_formatted_data(self, data: List[Dict[str, str]], output_path: str = "formatted_medical_data.json"):
        """Save formatted data to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved formatted data to {output_path}")
    
    def create_huggingface_dataset(self, data: List[Dict[str, str]]) -> Dataset:
        """Create a HuggingFace Dataset from formatted data"""
        dataset = Dataset.from_list(data)
        print(f"Created HuggingFace dataset with {len(dataset)} examples")
        return dataset
    
    def save_dataset(self, dataset: Dataset, output_dir: str = "medical_dialogue_dataset"):
        """Save dataset to disk"""
        dataset.save_to_disk(output_dir)
        print(f"Saved dataset to {output_dir}")
    
    def create_dataset_card(self, output_path: str = "dataset_card.md"):
        """Create a dataset card for documentation"""
        card_content = """---
language:
- en
license: mit
multilinguality:
- monolingual
size_categories:
- 10K<n<100K
source_datasets:
- original
task_categories:
- text-generation
- conversational
task_ids:
- dialogue-generation
- medical-assistant
---

# Medical Dialogue Dataset for Instruction Tuning

This dataset contains medical dialogues formatted for instruction tuning of language models.

## Dataset Description

- **Repository:** Medical Dialogue Dataset
- **Paper:** Medical dialogue generation for training medical assistants
- **Point of Contact:** [Your Contact]

### Dataset Summary

This dataset contains medical conversations between patients and doctors, formatted for instruction tuning of language models like Llama3.

### Supported Tasks and Leaderboards

- Medical dialogue generation
- Medical question answering
- Medical assistant training

### Languages

English

## Dataset Structure

### Data Instances

Each instance contains:
- `instruction`: The instruction for the model
- `input`: The conversation context
- `output`: The expected response (doctor's response)

### Data Fields

- `instruction` (string): Instruction for the model
- `input` (string): Patient's query and context
- `output` (string): Doctor's response

### Data Splits

- Training set: All available dialogues

## Additional Information

### Dataset Curators

[Your Name/Organization]

### Licensing Information

[Your License]

### Citation Information

[Your Citation]
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(card_content)
        print(f"Created dataset card: {output_path}")

def main():
    """Main function to prepare the data"""
    preparator = MedicalDialogueDataPreparator()
    
    try:
        # Create training data (limit to 10000 samples for testing)
        print("Creating training data...")
        training_data = preparator.create_training_data(max_samples=10000)
        
        if not training_data:
            print("No training data created. Please check your CSV file.")
            return
        
        # Save formatted data
        preparator.save_formatted_data(training_data)
        
        # Create HuggingFace dataset
        dataset = preparator.create_huggingface_dataset(training_data)
        
        # Save dataset
        preparator.save_dataset(dataset)
        
        # Create dataset card
        preparator.create_dataset_card()
        
        print("Data preparation completed!")
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()