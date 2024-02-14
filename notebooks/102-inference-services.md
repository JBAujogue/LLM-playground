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


Open a CLI at the root of the project, and run the service with :
```
uvicorn scripts.services.simple-llm-fastapi-service:app --root-path . --host 0.0.0.0 --port 8000
```


## 1.1 Direct requests to FastAPI service

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

## 1.2 Huggingface client around FastAPI service

```python
from huggingface_hub import InferenceClient
```

```python
InferenceClient
```

```python

```

```python

```

# 2. vLLM

```python

```

```python

```

# 3. Huggingface Text Generation Inference (TGI)


## 3.1 Simple client around TGI service

```python

```

## 3.2 Huggingface client around TGI service

See [Huggingface's official python client](https://github.com/huggingface/text-generation-inference/tree/main/clients/python) around TGI services, and [this example](https://towardsdatascience.com/llms-for-everyone-running-the-huggingface-text-generation-inference-in-google-colab-5adb3218a137).

```python

```

## 3.3 Langchain client around TGI service

See the [langchain community official doc](https://api.python.langchain.com/en/latest/llms/langchain_community.llms.huggingface_text_gen_inference.HuggingFaceTextGenInference.html#langchain-community-llms-huggingface-text-gen-inference-huggingfacetextgeninference), and [this example](https://towardsdatascience.com/llms-for-everyone-running-the-huggingface-text-generation-inference-in-google-colab-5adb3218a137).

```python
# see https://github.com/docker/docker-py
import docker
client = docker.from_env()
```

```python
image = 'ghcr.io/huggingface/text-generation-inference:1.4'
volumes = ['./checkpoints:/checkpoints']

client.containers.run(image, volumes = volumes)
```

```python
help(client.containers.run)
```

```python
# Basic Example (no streaming)
llm = HuggingFaceTextGenInference(
    inference_server_url="http://localhost:8010/",
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
)
print(llm("What is Deep Learning?"))

# Streaming response example
from langchain_community.callbacks import streaming_stdout

callbacks = [streaming_stdout.StreamingStdOutCallbackHandler()]
llm = HuggingFaceTextGenInference(
    inference_server_url="http://localhost:8010/",
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
    callbacks=callbacks,
    streaming=True
)
print(llm("What is Deep Learning?"))
```
