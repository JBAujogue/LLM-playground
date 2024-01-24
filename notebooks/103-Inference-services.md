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

<!-- #region -->


### Packages
<!-- #endregion -->

```python
%load_ext autoreload
%autoreload 2
```

```python
import os, sys
import time
import requests
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

chat_messages = [[
        dict(role = "system", content = f'{system}\n{context}'),
        dict(role = "user", content = query),
    ]
    for context, query in zip(context_list, query_list)
]
```

# 1. Simple FastAPI service

Run the service with :
```
uvicorn scripts.services.simple-llm-fastapi-service:app --root-path . --host 0.0.0.0 --port 8000
```

```python
url = 'http://localhost:8000/predict'

generation_params = dict(
    max_new_tokens = 128,
    num_beams = 1,
    batch_size = 1,
    do_sample = True,
    temperature = 0.7,
    top_p = 0.95,
    top_k = 40,
    repetition_penalty = 1.1,
)

resp = requests.post(
    url = url,
    json = dict(
        chat_messages = chat_messages,
        generation_params = generation_params,
    ),
)
resp.status_code
```

```python
resp.json()
```

```python

```

# 2. vLLM

```python

```

```python

```

# 3. TGI: Text Generation Inference
