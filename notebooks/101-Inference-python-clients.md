---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

In order to run a LLM on GPU locally, we use a quantized version of [zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta), which is a 7B parameter model belonging to the `Mistral-7B` lineage and originaly released by the Huggingface team (see their [blog post](https://www.unite.ai/zephyr-7b-huggingfaces-hyper-optimized-llm-built-on-top-of-mistral-7b/)), and was leading the Huggingface open LLM leaderboard as of October 2023.<br>
This model was subsequently quantized into [this model](https://huggingface.co/TheBloke/zephyr-7B-beta-GPTQ) using the [GPTQ algorithm](https://arxiv.org/pdf/2210.17323.pdf).

### Packages

```python
%load_ext autoreload
%autoreload 2
```

```python
import os, sys
import time
import copy

import pandas as pd
import torch
```

```python
device_int = int(torch.cuda.is_available())-1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_int, device
```

### Global variables

```python
model_id = 'TheBloke/zephyr-7B-beta-GPTQ'
model_id_gguf = 'TheBloke/zephyr-7B-beta-GGUF'

# model_id = 'TheBloke/openchat-3.5-0106-GPTQ'
```

```python
system = '''
    Answer the question based on the context below. Keep your answer short. 
    Only use information mentioned in context to form your answer.
    Respond "Unsure about answer" if not sure about the answer.

    Context:
    '''

context_list = [
    'Your name is Jean-baptiste',
    'The best year ever is 1986',
]

query_list = [
    "Comment tu t'appelle ?",
    'What is the best year of all time ?',
]

chat_messages = [[
        dict(role = "system", content = f'{system}\n{context}'),
        dict(role = "user", content = query),
    ]
    for context, query in zip(context_list, query_list)
]
```

# 1. Huggingface `transformers`

- Loading a `GPTQ`-quantized model rely on the [auto-gptq](https://huggingface.co/docs/transformers/quantization?bnb=8-bit#autogptq) and [optimum](https://huggingface.co/docs/optimum/index) packages
- Using the `device_map` keyword rely on the [accelerate](https://huggingface.co/docs/transformers/pipeline_tutorial#using-pipeline-on-large-models-with--accelerate-) package


## 1.1 Huggingface `pipeline`

```python
from transformers import pipeline, TextStreamer
```

```python
pipeline_config = dict(
    task = 'text-generation', 
    model = model_id, 
    device_map = device,
)

llm = pipeline(**pipeline_config)
```

```python
messages = [
    llm.tokenizer.apply_chat_template(m, tokenize = False, add_generation_prompt = True)
    for m in chat_messages
]
```

Generation arguments are provided at https://huggingface.co/docs/transformers/main_classes/text_generation .

```python
%%time

generation_params = dict(
    streamer = TextStreamer(llm.tokenizer, skip_prompt = True, skip_special_tokens = True),
    max_new_tokens = 256,
    num_beams = 1,
    batch_size = 1,
    do_sample = True,
    temperature = 0.7,
    top_p = 0.95,
    top_k = 40,
    repetition_penalty = 1.1,
)

answers = llm(messages, **generation_params)
```

```python
for a in answers:
    print(a[0]["generated_text"])
    print('---')
```

## 1.2 Huggingface `tokenizer` & `model`

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
```

```python
model_config = dict(
    pretrained_model_name_or_path = model_id, 
    device_map = device,
)

tokenizer = AutoTokenizer.from_pretrained(**model_config)
model = AutoModelForCausalLM.from_pretrained(**model_config)
```

```python
messages = [
    tokenizer.apply_chat_template(m, tokenize = False, add_generation_prompt = True)
    for m in chat_messages
]
```

We avoid the use of `streamer` since it is incompatible with input batching.

```python
%%time

generation_params = dict(
    streamer = None,
    max_new_tokens = 256,
    num_beams = 1,
    do_sample = True,
    temperature = 0.7,
    top_p = 0.95,
    top_k = 40,
    repetition_penalty = 1.1,
)

tokens = tokenizer(messages, padding = True, truncation = True, return_tensors = 'pt').input_ids.cuda()
tensors = model.generate(tokens, **generation_params)
answers = tokenizer.batch_decode(tensors)
```

```python
for a in answers:
    print(a)
    print('---')
```

## 1.3 `ctransformers`

- `ctransformers` makes faster CPU inference speed compared to base `transformers`, but support is limited to base or GGUF-quantized models.
- it also runs on GPU when installed with `ctransformers[cuda]` (although in this case it is best to go with `transformers` as it is not limited to GGUF quantization).

```python
from ctransformers import AutoModelForCausalLM
from transformers import AutoTokenizer, pipeline, TextStreamer
```

```python
model = AutoModelForCausalLM.from_pretrained(model_id_gguf, hf = True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

```python
pipeline_config = dict(
    task = 'text-generation', 
    tokenizer = tokenizer,
    model = model, 
    device_map = device,
)

llm = pipeline(**pipeline_config)
```

```python
messages = [
    llm.tokenizer.apply_chat_template(m, tokenize = False, add_generation_prompt = True)
    for m in chat_messages
]
```

```python
%%time

generation_params = dict(
    streamer = TextStreamer(llm.tokenizer, skip_prompt = True, skip_special_tokens = True),
    max_new_tokens = 256,
    batch_size = 1,
)

answers = llm(messages, **generation_params)
```

<!-- #region -->
## 1.4 `llama-index` wrapper

See `llama-index` [documentation](https://docs.llamaindex.ai/en/stable/module_guides/models/llms.html#using-llms) for instanciating models through its Huggingface wrapper.


- Support for text completion and chat endpoints (details below)
- Support for streaming and non-streaming endpoints
- Support for synchronous and asynchronous endpoints
<!-- #endregion -->

```python
import torch
from llama_index.llms import HuggingFaceLLM, ChatMessage
```

```python
llm_config = dict(
    tokenizer_name = model_id,
    model_name = model_id, 
    device_map = device_int,
    max_new_tokens = 256, 
)

llm = HuggingFaceLLM(**llm_config)
```

```python
torch.cuda.memory_allocated()
```

- `chat`
- `complete`
- `stream_chat`
- `stream_complete`
- `achat`
- `acomplete`
- `astream_chat`
- `astream_complete`

```python
messages = [[ChatMessage(**d) for d in m] for m in chat_messages]
```

```python
%%time

generation_params = dict(
    do_sample = True,
    temperature = 0.7,
    top_p = 0.95,
    top_k = 40,
    max_new_tokens = 256,
    repetition_penalty = 1.1,
    num_beams = 1,
)

answers = [llm.chat(ms, **generation_params) for ms in messages]
```

```python
for a in answers:
    print(a.dict()['message']['content'])
    print('---')
```

```python
%%time

answers_complete = [llm.complete(ms) for ms in messages]
```

```python
for a in answers_complete:
    print(a.dict()['text'])
    print('---')
```

## 1.X `langchain` wrapper

```python
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate


llm_config = dict(
    task = "text-generation",
    model_id = model_id, 
    #device = int(torch.cuda.is_available())-1,
    max_new_tokens = 256, 
    num_beams = 1, 
)

llm = HuggingFacePipeline.from_model_id(**llm_config)




template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)





%%time

chain = prompt | llm

question = "What is electroencephalography?"

print(chain.invoke({"question": question}))







from langchain.llms.huggingface_pipeline import HuggingFacePipeline

hf = HuggingFacePipeline.from_model_id(
    model_id="microsoft/DialoGPT-medium", task="text-generation", pipeline_kwargs={"max_new_tokens": 200, "pad_token_id": 50256},
)

from langchain.prompts import PromptTemplate

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

chain = prompt | hf

question = "What is electroencephalography?"

print(chain.invoke({"question": question}))
```
