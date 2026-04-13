# Financial Long Chain-of-Thought Compression Baseline Project

This project is built around the Qwen3-4B language model and targets financial/accounting question answering tasks. It includes data loading, model inference, result evaluation, and reinforcement learning training scripts.

## Project Structure

- `code/`
  - `main.py`: Main entry point that loads input data, builds prompts, runs batch inference, and saves results.
  - `data_loader.py`: Data loading and prompt preprocessing module that reads questions from CSV/TSV files.
  - `model_inference.py`: Model loading and inference module that supports `Qwen3-4B` and local model paths.
  - `evaluator.py`: Evaluation module that extracts answers, computes accuracy, measures CoT length, and scores outputs.
  - `Grpo.py`: GRPO training script based on TRL, including reward function, model loading, and training entry point.
  - `RLHF.py`: PPO-based RLHF script for reinforcement learning training.
  - `test.py`: Test/example script to verify model inference and prompt logic.
  - `train_grpo.py`: Example script for training with GRPO, including more advanced reward settings and logging.
- `env.yaml`: Conda environment configuration.
- `requirements.txt`: Python dependency list.
- `input.json`: Example input file in the project root.
- `output.csv`: Example/default output file.
- `code/data/`: Data directory containing training and inference datasets such as `input.csv`, `submit_example.csv`, and `dianjin_data`.
- `code/lora_model/`, `code/sft_final/`: Directories for LoRA weights and fine-tuned SFT models.

## Quick Start

### 1. Install Environment

Recommended using the provided conda environment:

```bash
conda env create -f env.yaml
conda activate qw3
```

Alternatively, install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

- Default input path: `code/data/input.csv`
- Expected format: TSV file. The current loader reads question text from the third column.
- Reference answers file format: TSV or CSV with `id` and `answer` columns.

### 3. Inference Workflow

Run the main script for batch inference:

```bash
python code/main.py \
  --input_path code/data/input.csv \
  --output_path ./output.csv \
  --reference_path code/data/submit_example.csv \
  --model_name_or_path <model_path_or_hf_name> \
  --max_new_tokens 4096 \
  --temperature 0.7 \
  --top_p 0.9 \
  --num_samples 5
```

Key arguments:

- `--input_path`: Input question file path
- `--output_path`: Output file path for model results
- `--reference_path`: Reference answer path for evaluation
- `--model_name_or_path`: Model name or local path
- `--system_prompt`: System prompt that instructs concise reasoning and final answer format
- `--num_samples`: Number of samples generated per question

### 4. Evaluate Results

If `--reference_path` is provided, `code/main.py` will run evaluation using `Evaluator`. The evaluator computes:

- extracted answers from model outputs
- accuracy
- shortest CoT length for each question
- final score based on negative total length

### 5. Reinforcement Learning Training

#### GRPO Training

Use `code/Grpo.py` for GRPO-based RL training:

```bash
python code/Grpo.py \
  --model_name_or_path <base_model_path> \
  --dataset_path <jsonl_dataset_path> \
  --lora_path <lora_weight_path> \
  --output_dir ./output/grpo
```

This script includes:

- reward function encouraging correctness and conciseness
- LoRA / PEFT model loading logic
- GRPOTrainer training flow

#### PPO / RLHF Training

`code/RLHF.py` provides PPO-based RLHF training logic, including:

- `load_sft_model()`: load a base model or PEFT model
- `prepare_for_ppo()`: wrap the model with a value head
- `compute_reward()`: compute rewards from answer correctness and conciseness
- `rlhf_training()`: training loop and model saving

### 6. Code Summary

- `DataLoader`: reads TSV data, extracts questions, builds prompts, and saves results
- `ModelInference`: loads the model/tokenizer, performs single and batch generation, and extracts answers
- `Evaluator`: loads output/reference files, extracts answers, computes accuracy, and scores predictions
- `Grpo.py`: RL training entry point with `reward_fn` and `grpo_training()`
- `RLHF.py`: PPO RLHF entry point with reward calculation and generation logic

## Notes

- The repository includes both Windows and Linux example paths. Adjust `--model_name_or_path`, `--input_path`, `--dataset_path`, and `--lora_path` for your environment.
- The current `code/data/input.csv` loader expects the question text in column 3. Update `DataLoader.get_questions()` if your data format differs.
- Use a small sample first for inference to verify model loading and tokenization before running on full datasets.

## Best Practices

- Run large models like Qwen3-4B on GPU when possible
- Use `use_4bit` options in `Grpo.py` / `RLHF.py` to reduce memory usage
- If local LoRA/SFT weights are available, set `--model_name_or_path` to the local directory

---

This README is based on the current project code. Add further details for data format or training steps as needed in future updates.
