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
from transformers import set_seed
```

```python
device_int = int(torch.cuda.is_available())-1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_int, device
```

### Global variables

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
```

```python
# --------- zephyr 7B -------------
zephyr7b_model_id = 'TheBloke/zephyr-7B-beta-GPTQ'
zephyr7b_model_id_gguf = 'TheBloke/zephyr-7B-beta-GGUF'
zephyr7b_chat_messages = [[
        dict(role = "system", content = f'{system}\n{context}'),
        dict(role = "user", content = query),
    ]
    for context, query in zip(context_list, query_list)
]
```

```python
# model_id = 'TheBloke/openchat-3.5-0106-GPTQ'
```

```python
# --------- mistral instruct -------------
mistral_instruct_model_id = 'TheBloke/Mistral-7B-Instruct-v0.2-GPTQ'
mistral_instruct_chat_messages = [[
        dict(role = "user", content = f'{system}\n{context}\n\n\n{query}'),
    ]
    for context, query in zip(context_list, query_list)
]
```

```python
# --------- zephyr 3B -------------
zephyr3b_model_id = 'stabilityai/stablelm-zephyr-3b'
zephyr3b_chat_messages = [[
        dict(role = "user", content = query),
    ]
    for context, query in zip(context_list, query_list)
]
```

Pick up a choice here:

```python
model_id = zephyr7b_model_id
chat_messages = zephyr7b_chat_messages
```

# 1. Huggingface transformers

- Loading a `GPTQ`-quantized model rely on the [auto-gptq](https://huggingface.co/docs/transformers/quantization?bnb=8-bit#autogptq) and [optimum](https://huggingface.co/docs/optimum/index) packages
- Using the `device_map` keyword rely on the [accelerate](https://huggingface.co/docs/transformers/pipeline_tutorial#using-pipeline-on-large-models-with--accelerate-) package

For inference, a LLM uses a template under the hood in order to aggregate instructions, context and past utterences into a single string. When calling this aggregation step, it is important to set `add_generation_prompt = True`, so that the final string is completed on the right with keywords marking the begining of a LLM's answer, thus engaging it to generate a response. See the transformers [documentation](https://huggingface.co/docs/transformers/main/chat_templating#how-do-i-use-chat-templates) about how to use chat templates.


## 1.1 Pipeline

```python
from transformers import pipeline, TextStreamer
```

```python
pipeline_config = dict(
    task = 'text-generation', 
    model = model_id, 
    device_map = 'auto',
    trust_remote_code = True,
)
set_seed(42)

llm = pipeline(**pipeline_config)
```

Generation arguments are provided [here](https://huggingface.co/docs/transformers/main_classes/text_generation).

```python
%%time

generation_params = dict(
    streamer = TextStreamer(llm.tokenizer, skip_prompt = True, skip_special_tokens = True),
    max_new_tokens = 128,
    num_beams = 1,
    batch_size = 1,
    do_sample = True,
    temperature = 0.2,
    top_p = 0.9,
    top_k = 40,
    repetition_penalty = 1.1,
)

answers = llm(chat_messages, **generation_params)
```

Note that input conversations are **not batched during inference**, but processed sequentially.

```python
for a in answers:
    print(a[0]["generated_text"][-1])
    print('---')
```

## 1.2 Tokenizer & model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
```

```python
model_config = dict(
    pretrained_model_name_or_path = model_id, 
    device_map = 'auto',
    trust_remote_code = True,
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

inputs = tokenizer(messages, padding = True, truncation = True, return_tensors = 'pt').input_ids
tensors = model.generate(inputs.to(model.device), **generation_params)
answers = tokenizer.batch_decode(tensors)
```

Note that input conversations are **batched during inference**, and therefore answers are consequently batched and padded.

```python
for a in answers:
    print(a)
    print('---')
```

# 2. ctransformers

- `ctransformers` makes faster CPU inference speed compared to base `transformers`, but support is limited to base or GGUF-quantized models.
- It is by now unmaintained.

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
# 3 llama-index

## 3.1 Huggingface wrapper

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

# 4. langchain

## 4.1 Huggingface wrapper

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
