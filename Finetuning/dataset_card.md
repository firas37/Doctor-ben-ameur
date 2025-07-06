---
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
