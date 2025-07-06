# Medical Assistant Llama3 Finetuning with Unsloth

This project implements finetuning of Llama3-8B model for medical dialogue assistance using Unsloth optimizations. The pipeline includes data preparation, model finetuning, and inference testing.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install requirements
pip install -r requirements.txt

# Or install Unsloth directly
pip install unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
```

### 2. Run the Complete Pipeline

```bash
python run_finetuning.py
```

This will:
- Check your environment (GPU, dependencies, data)
- Prepare the medical dialogue dataset
- Finetune Llama3-8B with Unsloth optimizations
- Test the model with example queries

### 3. Test Your Model

```bash
# Interactive chat
python inference.py --mode interactive

# Test with example queries
python inference.py --mode test
```

## 📁 Project Structure

```
Finetuning/
├── requirements.txt              # Dependencies
├── data_preparation.py          # Data formatting script
├── finetune_llama3.py          # Main finetuning script
├── inference.py                 # Model inference and testing
├── run_finetuning.py           # Complete pipeline runner
├── README.md                   # This file
├── medical_dialogue_dataset/   # Prepared dataset (created)
├── medical_llama3_finetuned/   # Finetuned model (created)
├── formatted_medical_data.json # Formatted training data (created)
└── inference_results.json      # Test results (created)
```

## 🔧 Configuration

### Model Configuration

The finetuning uses the following configuration:

- **Base Model**: `unsloth/llama-3-8b-bnb-4bit`
- **Quantization**: 4-bit quantization for memory efficiency
- **LoRA**: Rank 16, targeting attention and MLP layers
- **Sequence Length**: 2048 tokens
- **Learning Rate**: 2e-4
- **Batch Size**: 2 (per device)
- **Gradient Accumulation**: 4

### Data Format

The training data is formatted as instruction-following pairs:

```json
{
  "instruction": "You are a helpful medical assistant. Please respond to the patient's query in a professional and empathetic manner.",
  "input": "Patient: I have been experiencing headaches for the past week.",
  "output": "I understand you've been experiencing headaches for the past week. This could be due to various factors such as stress, dehydration, eye strain, or underlying medical conditions. I recommend..."
}
```

## 🛠️ Manual Steps

### Step 1: Data Preparation

```bash
python data_preparation.py
```

This script:
- Loads the medical dialogue dataset from `../datacreation/dialogues.csv`
- Formats conversations into instruction-following format
- Creates a HuggingFace dataset
- Saves the prepared data

### Step 2: Model Finetuning

```bash
python finetune_llama3.py
```

This script:
- Loads Llama3-8B with Unsloth optimizations
- Applies LoRA adapters for efficient finetuning
- Trains on the medical dialogue dataset
- Saves the finetuned model

### Step 3: Model Testing

```bash
python inference.py --mode test
```

This script:
- Loads the finetuned model
- Tests with example medical queries
- Saves results to `inference_results.json`

## 🎯 Advanced Usage

### Custom Training Parameters

Edit `finetune_llama3.py` to modify training parameters:

```python
finetuner.setup_training_arguments(
    num_train_epochs=5,           # More epochs
    per_device_train_batch_size=4, # Larger batch size
    learning_rate=1e-4,           # Different learning rate
    warmup_steps=200,             # More warmup steps
)
```

### Different Model Sizes

For different Llama3 variants:

```python
# Llama3-70B (requires more VRAM)
model_name = "unsloth/llama-3-70b-bnb-4bit"

# Llama3-1B (faster training, less VRAM)
model_name = "unsloth/llama-3-1b-bnb-4bit"
```

### Custom Data Format

Modify `data_preparation.py` to use different data formats:

```python
def format_for_chat(self, row):
    # Custom formatting logic
    return {
        "instruction": "Your custom instruction",
        "input": "Your custom input format",
        "output": "Your custom output"
    }
```

## 📊 Monitoring Training

The training process is logged to Weights & Biases. You can monitor:

- Training loss
- Evaluation loss
- Learning rate
- GPU utilization
- Memory usage

To disable WandB logging, comment out the `wandb.init()` call in `finetune_llama3.py`.

## 🔍 Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Reduce batch size: `per_device_train_batch_size=1`
   - Reduce sequence length: `max_seq_length=1024`
   - Use smaller model: `llama-3-1b-bnb-4bit`

2. **Slow Training**
   - Ensure GPU is available
   - Check if CUDA is properly installed
   - Verify Unsloth installation

3. **Data Loading Issues**
   - Check if `../datacreation/dialogues.csv` exists
   - Verify data format matches expected structure

4. **Model Loading Issues**
   - Check internet connection for model download
   - Verify sufficient disk space
   - Ensure all dependencies are installed

### Performance Tips

- Use a GPU with at least 16GB VRAM for optimal performance
- SSD storage recommended for faster data loading
- Close other applications to free up GPU memory
- Use gradient checkpointing for memory efficiency

## 📈 Expected Results

After successful finetuning, you should see:

- Training loss decreasing over epochs
- Model generating relevant medical responses
- Improved performance on medical dialogue tasks
- Professional and empathetic tone in responses

## 🤝 Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for efficient finetuning optimizations
- [Hugging Face](https://huggingface.co/) for the Transformers library
- [Meta](https://ai.meta.com/llama/) for the Llama3 model

## 📞 Support

For questions or issues:

1. Check the troubleshooting section above
2. Review the error messages carefully
3. Ensure all dependencies are correctly installed
4. Verify your GPU and CUDA setup

---

**Note**: This is a research project. The finetuned model should not be used for actual medical diagnosis or treatment without proper validation and medical oversight. 