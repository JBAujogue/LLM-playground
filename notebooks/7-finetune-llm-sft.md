---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region -->
References:
- The model we finetune comes from Microsoft's Github project [PyCodeGPT](https://github.com/microsoft/PyCodeGPT), and is available on the Huggingface model hub [here](https://huggingface.co/Daoguang/PyCodeGPT).
- Example [blog post](https://towardsdatascience.com/fine-tune-your-own-llama-2-model-in-a-colab-notebook-df9823a04a32) from Maxime Labonne
- Example [blog post](https://adithyask.medium.com/a-beginners-guide-to-fine-tuning-mistral-7b-instruct-model-0f39647b20fe) on finetuning on code generation (see this [benchmark](https://github.com/huybery/Awesome-Code-LLM) of pretrained coding assistants).
- Example [blog post](https://saankhya.medium.com/mistral-instruct-7b-finetuning-on-medmcqa-dataset-6ec2532b1ff1) on finetuning GPTQ-quantized model on medical QA
- Example [project](https://github.com/neuralwork/instruct-finetune-mistral/tree/main)
- [Finetune Your Own Llama 2 Model in a Colab Notebook](https://towardsdatascience.com/fine-tune-your-own-llama-2-model-in-a-colab-notebook-df9823a04a32)
- [Finetuning Mistral 7B Model with Your Custom Data](https://python.plainenglish.io/intruct-fine-tuning-mistral-7b-model-with-your-custom-data-7eb22921a483)
- [trl code snippet for SFT](https://github.com/huggingface/trl?tab=readme-ov-file#sfttrainer)
- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)




### Packages
<!-- #endregion -->

```python
%load_ext autoreload
%autoreload 2
```

```python
import os, sys
from pathlib import Path

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GPTQConfig,
    TrainingArguments,
    TextStreamer,
    set_seed,
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer
```

### Global variables

```python
path_to_repo = Path(os.getcwd()).parent
path_to_src  = os.path.join(path_to_repo, 'src')
path_to_data = os.path.join(path_to_repo, 'data')
path_to_runs = os.path.join(path_to_repo, 'mlruns')
```

```python
dataset_name = 'iamtarun/python_code_instructions_18k_alpaca'

input_model_name = 'TheBloke/Mistral-7B-Instruct-v0.2-GPTQ'
output_model_name = 'mistral-7b-instruct-v0.2-gptq--18k-alpaca'
```

```python
path_to_exp = os.path.join(path_to_runs, output_model_name)
path_to_exp
```

# 1. Prepare dataset

The instruction format to use is model-dependant. Fortunately the tokenizer should carry a pre-defined template in its `chat_template` attribute, for instance when it was already instruction-tuned. In this case it is advised to keep using this template for subsequent finetuning, by calling the `apply_chat_template` method of the tokenizer. See the transformers [chat templating documentation](https://huggingface.co/docs/transformers/main/chat_templating) for details. If no template was set in the tokenizer than a default one is shipped, although you are free of using whatever new template you like by setting up a new one as described in this page [edit chat templates](https://huggingface.co/docs/transformers/main/chat_templating#advanced-adding-and-editing-chat-templates).

```python
def create_text_rows(examples, tokenizer):
    '''
    Create prompts using model's tokenizer template.
    '''
    messages_list = [
        [
            {"role": "user", "content": f' {inst} Here are the inputs {inp} '},
            {"role": "assistant", "content": f' {out} '},
        ]
        for inst, inp, out in zip(examples['instruction'], examples['input'], examples['output'])
    ]
    return dict(text = [
        tokenizer.apply_chat_template(
            conversation = messages, tokenize = False, add_generation_prompt = False
        )
        for messages in messages_list
    ])
```

```python
tokenizer = AutoTokenizer.from_pretrained(input_model_name)
```

```python
# load dataset
dataset = load_dataset(dataset_name, split = 'train')

# create prompts within a new 'text' column
dataset_with_texts = dataset.map(
    function = lambda examples: create_text_rows(examples, tokenizer),
    batched = True,
)

# split into 80% train / 10% valid / 10% test
dataset_train_else = dataset_with_texts.train_test_split(test_size = .2, seed = 42, shuffle = False)
dataset_valid_test = dataset_train_else['test'].train_test_split(test_size = .5, seed = 42, shuffle = False)

# convert to dict of datasets
dataset_dict = dict(
    train = dataset_train_else['train'],
    valid = dataset_valid_test['train'],
    test  = dataset_valid_test['test'],
    all   = dataset_with_texts,
)
```

# 2. Select finetuning protocol

You can track metrics in tensorboard as finetuning proceeds: After opening a conda interpreter, moving to the repo's root directory and activating the environment, Tensorboard can be run with the command
```
tensorboard --logdir=mlruns
```


### Load tokenizer and model

Appropriately choosing the padding token and why it is important is discussed in this [issue](https://github.com/huggingface/transformers/issues/22794#issuecomment-1598977285), see this alternative [solution](https://medium.com/@mayvic/solving-the-issue-of-falcon-text-generation-never-stopping-e8f599eae8f0) as well.

```python
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(input_model_name)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.unk_token 

# Load base model
quantization_config = dict(bits = 4, use_exllama = False)
quantization_config = GPTQConfig(**quantization_config)

model_config = dict(
    pretrained_model_name_or_path = input_model_name,
    quantization_config = quantization_config,
    device_map = 'auto',
)

model = AutoModelForCausalLM.from_pretrained(**model_config)
```

Run some dummy inference prior training:

```python
generation_params = dict(
    streamer = TextStreamer(tokenizer, skip_prompt = True, skip_special_tokens = True),
    max_new_tokens = 64,
    num_beams = 1,
    do_sample = True,
    temperature = 0.2,
    top_p = 0.9,
    top_k = 40,
    repetition_penalty = 1.1,
)
set_seed(42)
model.eval()
with torch.no_grad():
    message = [{"role": "user", "content": "Write a function that prints 'hello world' in python"}]
    message = tokenizer.apply_chat_template(message, tokenize = False, add_generation_prompt = True)
    inputs = tokenizer(message, return_tensors = 'pt').to(model.device)
    tensors = model.generate(**inputs, **generation_params)
    answers = tokenizer.batch_decode(tensors)
```

### Train model

```python
model.config.use_cache = False
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
```

```python
peft_config = LoraConfig(
    task_type = "CAUSAL_LM",
    bias = "none",
    r = 64,
    lora_alpha = 16,
    lora_dropout = 0.05,
)

training_config = TrainingArguments(
    output_dir = path_to_exp,
    num_train_epochs = 1,
    per_device_train_batch_size = 4,
    per_device_eval_batch_size = 4,
    gradient_accumulation_steps = 1,
    gradient_checkpointing = True,
    optim = "adamw_torch",
    learning_rate = 3e-4,
    weight_decay = 1e-3,
    # fp16 = False,
    # bf16 = False,
    max_grad_norm = 0.3,
    max_steps = 300,
    warmup_ratio = 0.05,
    group_by_length = True,
    lr_scheduler_type = "cosine",
    save_strategy = "no",
    logging_steps = 25,
    logging_first_step = True,
    report_to = "tensorboard",
    seed = 42,
)

trainer_config = dict(
    max_seq_length = None,
    packing = False,
)

# Set supervised fine-tuning trainer
trainer = SFTTrainer(
    tokenizer = tokenizer,
    model = model,
    train_dataset = dataset_dict['train'],
    eval_dataset = dataset_dict['valid'],
    dataset_text_field = "text",
    peft_config = peft_config,
    args = training_config,
    **trainer_config,
)
```

```python
trainer.model.print_trainable_parameters()
```

```python
trainer.train()
```

### Run inference post training

```python
generation_params = dict(
    streamer = TextStreamer(tokenizer, skip_prompt = True, skip_special_tokens = True),
    max_new_tokens = 128,
    num_beams = 1,
    do_sample = True,
    temperature = 0.2,
    top_p = 0.9,
    top_k = 40,
    repetition_penalty = 1.1,
)

set_seed(42)
model.eval()
with torch.no_grad():
    message = [{"role": "user", "content": "Write a function that prints 'hello world' in python"}]
    message = tokenizer.apply_chat_template(message, tokenize = False, add_generation_prompt = True)
    inputs = tokenizer(message, return_tensors = 'pt').to(model.device)
    tensors = model.generate(**inputs, **generation_params)
    answers = tokenizer.batch_decode(tensors)
```

# 3. Finetune final model

After carefully selecting your finetuning hyperparameters through multiple training runs, it is advised to store your final finetuning parameters and run a final finetuning over the whole dataset, instead of only using the 'train' slice.


### Run finetuning

```python
# your turn here:
# - load base tokenizer and model
# - set trainer with your selected finetuning parameters
# - run finetuning
```

### Merge and serialize model

```python
# merge model and adapter
merged_model = trainer.model.merge_and_unload()
```

```python
# save merged model
merged_model.save_pretrained(os.path.join(path_to_exp, 'model'))
```

```python
tokenizer.padding_side = 'left'
tokenizer.save_pretrained(os.path.join(path_to_exp, 'tokenizer'))
```

### Load finetuned model

```python
# load tokenizer 
tokenizer = AutoTokenizer.from_pretrained(os.path.join(path_to_exp, 'tokenizer'))

# load finetuned model
finetuned_model = AutoModelForCausalLM.from_pretrained(
    os.path.join(path_to_exp, 'model'),
    device_map = 'auto',
).eval()
```

```python
generation_params = dict(
    streamer = TextStreamer(tokenizer, skip_prompt = True, skip_special_tokens = True),
    max_new_tokens = 128,
    num_beams = 1,
    do_sample = True,
    temperature = 0.2,
    top_p = 0.9,
    top_k = 40,
    repetition_penalty = 1.1,
)

set_seed(42)
finetuned_model.eval()
with torch.no_grad():
    message = [{"role": "user", "content": "Write a function that prints 'hello world' in python"}]
    message = tokenizer.apply_chat_template(message, tokenize = False, add_generation_prompt = True)
    inputs = tokenizer(message, return_tensors = 'pt').to(model.device)
    tensors = finetuned_model.generate(**inputs, **generation_params)
    answers = tokenizer.batch_decode(tensors)
```
