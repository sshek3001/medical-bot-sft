# SFT01 — Llama 3.2 1B QLoRA Fine-Tuning to create a healthcare bot

SFT01 is an experimental project for fine-tuning **Llama 3.2 1B Instruct** using **Supervised Fine-Tuning (SFT)** with **QLoRA**.

The project is being developed on a local machine with an NVIDIA RTX 4050 GPU with 6 GB VRAM.

## Overview

The training pipeline is:

```text
Training Dataset
      ↓
Conversational JSONL
      ↓
Llama Chat Template
      ↓
Tokenization
      ↓
QLoRA
      ├── 4-bit quantized base model
      └── trainable LoRA adapters
      ↓
SFTTrainer
      ↓
LoRA Adapter Checkpoints
```

The original Llama model is kept unchanged. Training updates only the LoRA adapter parameters.

## Base Model

The project uses the locally stored:

```text
Llama 3.2 1B Instruct
```

The base model is stored separately from the project:

```text
D:\models\llama-3.2-1b-instruct
```

This separation ensures that different experiments can use independent LoRA adapters without modifying the original model.

## Hardware

```text
GPU: NVIDIA RTX 4050
VRAM: 6 GB
```

The base model was tested in FP16 and used approximately:

```text
~2.3 GB VRAM
```

When loaded using 4-bit quantization, GPU memory usage dropped to approximately:

```text
~0.96 GB VRAM
```

This provides substantially more memory headroom for fine-tuning.

## QLoRA

QLoRA is implemented by combining:

1. **4-bit quantization** of the base model using `bitsandbytes`
2. **LoRA adapters** using PEFT

The 4-bit configuration is:

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)
```

The model is then loaded with:

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto"
)
```

The LoRA configuration uses:

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    task_type="CAUSAL_LM"
)
```

Only the LoRA parameters are trained while the quantized base model remains frozen.

In an initial experiment, the model contained approximately:

```text
Total parameters:      1.239B
Trainable parameters:  3.4M
Trainable percentage:  0.275%
```

## Dataset Format

Training data is stored in JSONL format:

```text
data/
└── train.jsonl
```

Each line represents one conversation.

The conversation uses the standard message structure:

```json
{
    "messages": [
        {
            "role": "system",
            "content": "..."
        },
        {
            "role": "user",
            "content": "..."
        },
        {
            "role": "assistant",
            "content": "..."
        }
    ]
}
```

The project initially used five manually created conversations to verify the SFT pipeline.

A public healthcare customer-service dataset containing 100 examples was subsequently converted from:

```text
system_prompt
prompt
completion
```

into the conversational `messages` format before training.

## Tokenization

The tokenizer associated with the base Llama model is loaded using:

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
```

The conversation is formatted using Llama's chat template:

```python
tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False
)
```

The resulting text is then tokenized into token IDs.

The project also contains `tokenize_data.py`, which was used to study:

- tokenization
- token IDs
- input sequences
- labels
- `-100` loss masking
- assistant-token training targets

For the actual SFT run, `SFTTrainer` performs the dataset preprocessing and label preparation internally.

## Supervised Fine-Tuning

Training is handled by Hugging Face TRL's:

```python
SFTTrainer
```

The trainer receives the conversational dataset directly:

```python
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    processing_class=tokenizer,
    peft_config=lora_config
)
```

The trainer handles the SFT preprocessing and training loop.

## Training Configuration

The initial experiment used:

```python
SFTConfig(
    output_dir="./outputs",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="epoch",
    fp16=False,
    report_to="none"
)
```

The batch size was kept at `1` to accommodate the 6 GB VRAM constraint.

Gradient accumulation allows multiple examples to contribute to an optimizer update without requiring the entire accumulated batch to reside in VRAM simultaneously.

## Training Results

An initial five-example experiment confirmed that the complete QLoRA pipeline worked.

A subsequent 100-example healthcare customer-service dataset was used for a larger experiment.

The training process successfully produced LoRA checkpoints at:

```text
outputs/
├── checkpoint-1/
├── checkpoint-2/
├── checkpoint-3/
├── checkpoint-13/
├── checkpoint-26/
└── checkpoint-39/
```

With three epochs and approximately 39 optimizer steps, the final checkpoint was:

```text
checkpoint-39
```

The checkpoints contain the LoRA adapter weights rather than a complete copy of the base model.

Important files include:

```text
adapter_model.safetensors
adapter_config.json
```

## Testing the Fine-Tuned Model

The trained adapter can be loaded on top of the original base model using PEFT.

The testing flow is:

```text
Original Llama 3.2 1B
        +
checkpoint-39 LoRA adapter
        ↓
Fine-tuned model
        ↓
User prompt
        ↓
Generated response
```

The original model remains untouched.

A separate testing script, `test_sft.py`, loads the quantized base model and then attaches the trained adapter:

```python
model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH
)
```

The fine-tuned model was successfully loaded and generated responses after training.


## Future Work

The next stage is to construct a dataset specifically aligned with the intended hospital-assistant task rather than relying primarily on a general healthcare customer-service dataset.

The intended model behavior includes:

```text
Patient request
      ↓
Understand complaint
      ↓
Identify appropriate department
      ↓
Select appropriate doctor
      ↓
Check availability
      ↓
Schedule appointment
      ↓
Confirm appointment
```

Dynamic information such as doctor availability and appointment slots should eventually come from external tools or databases rather than being memorized by the language model.

## Environment

The project uses the following major libraries:

```text
Python
PyTorch
Transformers
TRL
PEFT
bitsandbytes
datasets
```

The model and tokenizer are loaded from the local model directory rather than downloaded during training.