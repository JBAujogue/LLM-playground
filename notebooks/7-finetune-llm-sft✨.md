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

The two following projects are managed by Huggingface and are at the core of LLM finetuning:
- [TRL](https://github.com/huggingface/trl/tree/main/examples/scripts)
- [alignment-handbook](https://github.com/huggingface/alignment-handbook/tree/main/scripts)

The `alignment-handbook` project can serve as a model factory.

Other references:
- Huggingface Supervized Fine-Tuning [tutorial guide](https://huggingface.co/docs/trl/sft_trainer).
- W&B [tutorial guide](https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-tune-an-LLM-Part-3-The-HuggingFace-Trainer--Vmlldzo1OTEyNjMy).
- Confident-AI [tutorial guide](https://www.confident-ai.com/blog/the-ultimate-guide-to-fine-tune-llama-2-with-llm-evaluations#evaluating-a-fine-tuned-llm-with-deepeval).
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
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
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
output_model_name = 'mistral-7b-instruct-v0.2-gptq-lora--18k-alpaca'
```

```python
path_to_exp = os.path.join(path_to_runs, output_model_name)
path_to_exp
```

```python



```

# 1. Prepare dataset

Data items are converted to single string in a format which is model-dependant and which depends on the pre-defined template in the `chat_template` attribute of the tokenizer. If this chat template already exists, for instance when the model was already instruction-tuned, then it is advised to keep using it for futher finetuning. If no template was set in the tokenizer then a default one is shipped, although you are free of using whatever new template you like by setting up a new one as described in this page [edit chat templates](https://huggingface.co/docs/transformers/main/chat_templating#advanced-adding-and-editing-chat-templates). Some utility function making the creation of template straightforward is also described [here](https://huggingface.co/docs/trl/sft_trainer#add-special-tokens-for-chat-format).

 The `SFTTrainer` class natively supports formatting of datasets as list of chat messages, see the huggingface [sft tutorial](https://huggingface.co/docs/trl/sft_trainer#dataset-format-support), but calling the `apply_chat_template` by ourselves make the dataset seamlessly compatible with other finetuning features, such as [completion-only finetuning](https://huggingface.co/docs/trl/sft_trainer#train-on-completions-only).)

```python
def apply_chat_template(examples, tokenizer):
    '''
    Create prompts using model's tokenizer template.
    '''
    messages_list = [
        [
            dict(role = 'user', content = f' {inst} Here are the inputs {inp} '),
            dict(role = 'assistant', content = f' {out} '),
        ]
        for inst, inp, out in zip(
            examples['instruction'], examples['input'], examples['output']
        )
    ]
    return dict(text = [
        tokenizer.apply_chat_template(
            m, tokenize = False, add_generation_prompt = False
        )
        for m in messages_list
    ])
```

```python
tokenizer = AutoTokenizer.from_pretrained(input_model_name)
```

```python
# load dataset
dataset = load_dataset(dataset_name, split = 'train')

# convert to chat messages within a new 'messages' column
dataset_chat = dataset.map(
    function = lambda examples: apply_chat_template(examples, tokenizer),
    remove_columns = dataset.column_names,
    batched = True,
)
# split into 80% train / 10% valid / 10% test
dataset_train_else = dataset_chat.train_test_split(test_size = .2, seed = 42, shuffle = False)
dataset_valid_test = dataset_train_else['test'].train_test_split(test_size = .5, seed = 42, shuffle = False)

# convert to dict of datasets
dataset_dict = dict(
    train = dataset_train_else['train'],
    valid = dataset_valid_test['train'],
    test = dataset_valid_test['test'],
    all = dataset_chat,
)
```

# 2. Select finetuning protocol


### Load tokenizer and model

According to this [demo notebook](https://colab.research.google.com/drive/1_TIrmuKOFhuRRiTWN94iLKUFu6ZX4ceb?usp=sharing#scrollTo=l-EwnZd0jpwB):

> We disable the exllama kernel because training with exllama kernel is unstable. To do that, we pass a GPTQConfig object with disable_exllama=True. This will overwrite the value stored in the config of the model.


```python
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(input_model_name)

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
    message = [dict(role = 'user', content = "Write a function that prints 'hello world' in python")]
    message = tokenizer.apply_chat_template(message, tokenize = False, add_generation_prompt = True)
    inputs = tokenizer(message, return_tensors = 'pt').to(model.device)
    tensors = model.generate(**inputs, **generation_params)
    answers = tokenizer.batch_decode(tensors)
```

### Prepare tokenizer and model for training

We further set the padding side to the right for batched finetuning, and set a padding token to the tokenizer when it does not natively carry one.<br>
Appropriately choosing the padding token is important, as stated in Huggingface [SFT usage](https://huggingface.co/docs/trl/sft_trainer):

> Make sure to have a pad_token_id which is different from eos_token_id which can result in the model not properly predicting EOS (End of Sentence) tokens during generation.

See also this [issue](https://github.com/huggingface/transformers/issues/22794#issuecomment-1598977285) and this alternative [solution](https://medium.com/@mayvic/solving-the-issue-of-falcon-text-generation-never-stopping-e8f599eae8f0) as well.

```python
tokenizer.padding_side = "right"
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.unk_token 
```

```python
model.config.use_cache = False
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
```

### Prepare trainer

We perform Instruction masking using the [DataCollatorForCompletionOnlyLM](https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py) class from the `trl` library in order to make the loss only computed on tokens of the LLM answer. As quoted [here]():

>  Instruction masking consists in setting the token labels of the instructions to -100 (the default value that the CrossEntropy PyTorch function ignores). This way, you're only back-propagating on the completions.

See also the [completion-only finetuning](https://huggingface.co/docs/trl/sft_trainer#train-on-completions-only) tutorial.

```python
peft_config = LoraConfig(
    task_type = "CAUSAL_LM",
    bias = "none",
    r = 32,
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
    optim = "paged_adamw_32bit",
    learning_rate = 3e-4,
    weight_decay = 1e-3,
    fp16 = False,
    bf16 = False,
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
    neftune_noise_alpha = 5,
    packing = False,
    data_collator = DataCollatorForCompletionOnlyLM(response_template = '[/INST]', tokenizer = tokenizer),
    dataset_text_field = 'text',
)

# Set supervised fine-tuning trainer
trainer = SFTTrainer(
    tokenizer = tokenizer,
    model = model,
    peft_config = peft_config,
    train_dataset = dataset_dict['train'],
    eval_dataset = dataset_dict['valid'],
    args = training_config,
    **trainer_config,
)
```

```python
trainer.model.print_trainable_parameters()
```

Remark: The trainer has a `model` attribute wihich does _not_ correspond to the original model. Instead, the original model is contained into a sub-sub-subattribute of the trainer object:

```python
model == trainer.model.base_model.model
```

### Run training

You can track metrics in tensorboard as finetuning proceeds: After opening a conda interpreter, moving to the repo's root directory and activating the environment, Tensorboard can be run with the command
```
tensorboard --logdir=mlruns
```

```python
trainer.train()
```

### Compute test score

The `evaluate` method of a Trainer object does not work out of the box:

Calling `trainer.evaluate()` runs loss computation on the trainer's `eval_dataset` attribute, which was internally preprocessed and is described by the columns `['input_ids', 'attention_mask']`. In turns, calling `trainer.evaluate(dataset_dict['test'])` raises an error since the dataset passed as parameter is descivbed by the columns `['messages']`.

```python

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
    message = [dict(role = 'user', content = "Write a function that prints 'hello world' in python")]
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

### Save & load result

The trained adapter can be merged to the original model, and the result can be subsequently serialized to disk, as the following code snippet shows:

```python
# # merge model and adapter
# merged_model = trainer.model.merge_and_unload()

# # save merged model
# merged_model.save_pretrained(os.path.join(path_to_exp, 'model'))

# tokenizer.padding_side = 'left'
# tokenizer.save_pretrained(os.path.join(path_to_exp, 'tokenizer'))
```

However in order to save disk space and be able to plug and unplug it at will, we consider saving tha adapter only:

```python
# save merged model
trainer.model.save_pretrained(os.path.join(path_to_exp, 'model'))

tokenizer.padding_side = 'left'
tokenizer.save_pretrained(os.path.join(path_to_exp, 'tokenizer'))
```

### Run inference

```python
# re-load tokenizer & finetuned model
tokenizer = AutoTokenizer.from_pretrained(os.path.join(path_to_exp, 'tokenizer'))
finetuned_model = AutoModelForCausalLM.from_pretrained(
    os.path.join(path_to_exp, 'model'), device_map = 'auto',
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
    message = [dict(role = 'user', content = "Write a function that prints 'hello world' in python")]
    message = tokenizer.apply_chat_template(message, tokenize = False, add_generation_prompt = True)
    inputs = tokenizer(message, return_tensors = 'pt').to(finetuned_model.device)
    tensors = finetuned_model.generate(**inputs, **generation_params)
    answers = tokenizer.batch_decode(tensors)
```
