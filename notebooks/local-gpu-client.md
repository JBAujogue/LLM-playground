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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
```

```python
model_id = 'TheBloke/zephyr-7B-beta-GPTQ'
```

```python
context_list = [
    'Your name is Jean-baptiste',
    'The best year ever is 1986',
]

query_list = [
    '''Comment tu t'appelle ?''',
    'What is the best year of all time ?',
]
```

# 1. Huggingface


## 1.1 Huggingface `pipeline`

- Loading a `GPTQ`-quantized model rely on the [auto-gptq](https://huggingface.co/docs/transformers/quantization?bnb=8-bit#autogptq) and [optimum](https://huggingface.co/docs/optimum/index) packages
- Using the `device_map` keyword rely on the [accelerate](https://huggingface.co/docs/transformers/pipeline_tutorial#using-pipeline-on-large-models-with--accelerate-) package

```python
from transformers import pipeline
```

Generation arguments are provided at https://huggingface.co/docs/transformers/main_classes/text_generation.

```python
llm_config = dict(
    task = 'text-generation',
    model = model_id, 
    device_map = device,
    max_new_tokens = 256, 
    num_beams = 1, 
)

llm = pipeline(**llm_config)
```

```python
messages = [
    [{
        "role": "system",
        "content": 
        f'''
        Answer the question based on the context below. Keep your answer short. 
        Only use information mentioned in context to form your answer.
        Respond "Unsure about answer" if not sure about the answer.
        
        Context:
        {context}
        ''',
    },
    {
        "role": "user", 
        "content": query,
    },
    ]
    for context, query in zip(context_list, query_list)
]
prompts = [
    llm.tokenizer.apply_chat_template(m, tokenize = False, add_generation_prompt = True)
    for m in messages
]
```

```python
%%time

answers = llm(prompts)
```

```python
for a in answers:
    print(a[0]["generated_text"])
    print('---')
```

## 1.2 `langchain` wrapper

```python
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
```

```python
help(HuggingFacePipeline.from_model_id)
```

```python
llm_config = dict(
    task = "text-generation",
    model_id = model_id, 
    #device = int(torch.cuda.is_available())-1,
    max_new_tokens = 256, 
    num_beams = 1, 
)

llm = HuggingFacePipeline.from_model_id(**llm_config)
```

```python


template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)
```

```python
%%time

chain = prompt | llm

question = "What is electroencephalography?"

print(chain.invoke({"question": question}))
```

```python
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

```python

```

```python

```
