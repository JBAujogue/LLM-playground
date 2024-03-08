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
- Example [blog post](https://adithyask.medium.com/a-beginners-guide-to-fine-tuning-mistral-7b-instruct-model-0f39647b20fe) on finetuning on code generation
- Example [blog post](https://saankhya.medium.com/mistral-instruct-7b-finetuning-on-medmcqa-dataset-6ec2532b1ff1) on finetuning GPTQ-quantized model on medical QA
- Example [project](https://github.com/neuralwork/instruct-finetune-mistral/tree/main)

Unfortunately all examples provided here carry a quantization step prior to training, which is not possible as-is on CPU.
We therefore adapt this material to run finetuning of a "small" **unquantized** model with 110M parameters on CPU.


Reflections:
- The model is finetuned on the causal language modeling over the _full sequence of tokens_ carried by each training sample. This means that the model in trained not only to generate the expected answer of an instruction in english, but is also finetuned on the task of generating the instruction itself. This training objective can be suboptimal, particularly in this example as the base model is pre-trained to generate code only, but is now finetuned to generate english utterances as well. This can be avoided by crafting the appropriate `DataCollator`, which will make the loss function skip instruction tokens in each gradient descent loop. This is left for future development.
- It is encouraged to run this tutorial on a quantized 7B parameter model whenever the appropriate hardware becomes available.
- Should this process be applied on a client project, it is advised to convert this tutorial notebook into a training script, with all parameters cast in an external configuration file.

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
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    TextStreamer,
    set_seed,
)
from peft import LoraConfig, PeftModel
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

input_model_name = 'Daoguang/PyCodeGPT'
output_model_name = 'pycodegpt--18k-alpaca'
```

```python
path_to_exp = os.path.join(path_to_runs, output_model_name)
path_to_exp
```

# 1. Prepare dataset

The instruction format to use is model-dependant. If the model tokenizer does not carry a pre-defined template, as it is the case for the `PyCodeGPT` model we aim at finetuning, we are free of using whatever prompt template we want. If the model comes with a pre-defined template, for instance when it was already insteruction-tuned, then subsequent finetuning must be applied on text data formated using the same template. To find the appropriate template, we can either refer to the Huggingface model page (be careful as the snippet can be erroneous), or refer to the model's tokenizer, which often carry the template and can be used as shown below.

```python
def create_text_rows(examples):
    '''
    Create prompts using manually-defined template.
    Special tokens that are used must agree with those of the model's tokenizer.
    '''
    return dict(text = [
        f"""<s>[INST]{inst} Here are the inputs {inp}[/INST]{out}</s>"""
        for inst, inp, out in zip(examples['instruction'], examples['input'], examples['output'])
    ])


def create_text_rows_with_tokenizer_template(examples, tokenizer):
    '''
    Create prompts using model's tokenizer template.
    '''
    messages_list = [
        [
            {"role": "user", "content": f'{inst} Here are the inputs {inp}'},
            {"role": "assistant", "content": out},
        ]
        for inst, inp, out in zip(examples['instruction'], examples['input'], examples['output'])
    ]
    return dict(text = [
        tokenizer.apply_chat_template(messages, tokenize = False)
        for messages in messages_list
    ])
```

```python
tokenizer = AutoTokenizer.from_pretrained(input_model_name)
```

```python
# load dataset
dataset = load_dataset(dataset_name, split = 'train')
```

```python
dataset[0]
```

```python
# create prompts within a new 'text' column
# dataset_with_texts = dataset.map(create_text_rows, batched = True)
dataset_with_texts = dataset.map(
    function = lambda examples: create_text_rows_with_tokenizer_template(examples, tokenizer),
    batched = True,
)
```

```python
dataset_with_texts
```

```python
dataset_with_texts[0]
```

```python
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

```python
dataset_dict['valid'][0]
```

# 2. Select finetuning protocol

You can track metrics in tensorboard as finetuning proceeds: After opening a conda interpreter, moving to the repo's root directory and activating the environment, Tensorboard can be run with the command
```
tensorboard --logdir=mlruns
```


### Load tokenizer and model

```python
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(input_model_name)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.unk_token 

# appropriately choosing the padding token and why it is important is discussed in this issue:
# https://github.com/huggingface/transformers/issues/22794#issuecomment-1598977285
# see this alternative solution as well:
# https://medium.com/@mayvic/solving-the-issue-of-falcon-text-generation-never-stopping-e8f599eae8f0
```

```python
# Load base model
model_config = dict(
    pretrained_model_name_or_path = input_model_name,
    device_map = 'auto',
)

model = AutoModelForCausalLM.from_pretrained(**model_config)
model.config.use_cache = False
model.config.pretraining_tp = 1
```

Run some dummy inference prior training:

```python
message = [{"role": "user", "content": "Write a function that prints 'hello world' in python"}]
message = tokenizer.apply_chat_template(message, tokenize = False)
message
```

```python
generation_params = dict(
    streamer = TextStreamer(tokenizer, skip_prompt = True, skip_special_tokens = True),
    max_new_tokens = 32,
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
    inputs = tokenizer(message, return_tensors = 'pt').to(model.device)
    tensors = model.generate(**inputs, **generation_params)
    answers = tokenizer.batch_decode(tensors)
```

### Train model

```python
peft_config = LoraConfig(
    task_type = "CAUSAL_LM",
    bias = "none",
    r = 64,
    lora_alpha = 16,
    lora_dropout = 0.1,
)

training_config = TrainingArguments(
    output_dir = path_to_exp,
    num_train_epochs = 1,
    per_device_train_batch_size = 4,
    per_device_eval_batch_size = 4,
    gradient_accumulation_steps = 1,
    gradient_checkpointing = True,
    optim = "adamw_torch",
    learning_rate = 2e-4,
    weight_decay = 0.001,
    fp16 = False,
    bf16 = False,
    max_grad_norm = 0.3,
    max_steps = 300,
    warmup_ratio = 0.05,
    group_by_length = True,
    lr_scheduler_type = "linear",
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
trainer.model
```

```python
trainer.train()
```

### Run inference post training

```python
message = [{"role": "user", "content": "Write a function that prints 'hello world' in python"}]
message = tokenizer.apply_chat_template(message, tokenize = False)
message
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
model.eval()
with torch.no_grad():
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
tokenizer = AutoTokenizer.from_pretrained(os.path.join(path_to_exp, 'tokenizer'))
```

```python
finetuned_model = AutoModelForCausalLM.from_pretrained(
    os.path.join(path_to_exp, 'model'),
    device_map = 'auto',
).eval()
```

```python
message = [{"role": "user", "content": " Write a function that prints 'hello world' in python."}]
message = tokenizer.apply_chat_template(message, tokenize = False)

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
    inputs = tokenizer(message, return_tensors = 'pt').to(model.device)
    tensors = finetuned_model.generate(**inputs, **generation_params)
    answers = tokenizer.batch_decode(tensors)
```
