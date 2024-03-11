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
import uvicorn
import docker

# TGI
from text_generation import InferenceAPIClient, InferenceAPIAsyncClient
from huggingface_hub import InferenceClient, AsyncInferenceClient

from langchain_community.llms import HuggingFaceTextGenInference
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

from openai import OpenAI
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
uvicorn scripts.services.fastapi-service:app --root-path . --host 0.0.0.0 --port 8000
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

## 1.2 Huggingface-hub client around FastAPI service

```python

```

# 2. Text Generation Inference (TGI)

<!-- #region -->
Run the backend service with one of the following methods:

1. Start an existing container: open Docker desktop and run the appropriate container.

2. Create and run a new container: open a CLI at the root of the project, and run the command
```
wsl -e ./scripts/services/tgi-service.sh
```


3. Start an existing container with Docker's python API:
<!-- #endregion -->

```python
# see:
# - https://github.com/docker/docker-py
# - https://docker-py.readthedocs.io/en/stable/containers.html
container_id = '146dc7ff2d842b77c5bc173c53d1b2e1e4b02a0a5f96ea243307f44b30a773b6'

client = docker.from_env()
client.containers.get(container_id).start()
```

```python
url = 'http://127.0.0.1:8080'
```

### 2.1 Simple client around TGI service

See the TGI [docs](https://huggingface.co/docs/text-generation-inference/quicktour).

```python
headers = {"Content-Type": "application/json"}

generation_config = dict(
    temperature = 0.1,
    top_p = .1,
    max_tokens = 64,
    stream = True,
    seed = 42,
)

data = {
    'inputs': 'What is the most beautiful city in the world ? please elaborate on your answer. Do not share your name.',
    'parameters': generation_config,
}
response = requests.post(f'{url}/generate', headers = headers, json = data)
```

```python
response.json()['generated_text']
```

### 2.2 Text-generation client around TGI service

See [Huggingface's official python client](https://github.com/huggingface/text-generation-inference/tree/main/clients/python) around TGI services. **Not working on local endpoints**.

```python
# client = InferenceAPIClient(url)
# client = InferenceAPIAsyncClient(url)
```

### 2.3 Huggingface-hub client around TGI service

See [Huggingface-hub python client](https://huggingface.co/docs/text-generation-inference/basic_tutorials/consuming_tgi) around TGI services.

```python
client = InferenceClient(model = url)
```

```python
generation_config = dict(
    temperature = 0.1,
    top_p = .1,
    max_new_tokens = 64,
    stream = True,
    seed = 42,
)

response = client.text_generation("How do you make cheese ?", **generation_config)
```

```python
for token in response:
    print(token, end = '')
```

### 2.4 Langchain client around TGI service

See the [langchain community official doc](https://python.langchain.com/docs/integrations/llms/huggingface_endpoint) (also see this [deprecated page](https://api.python.langchain.com/en/latest/llms/langchain_community.llms.huggingface_text_gen_inference.HuggingFaceTextGenInference.html#langchain-community-llms-huggingface-text-gen-inference-huggingfacetextgeninference) in case of trouble), and [this example](https://towardsdatascience.com/llms-for-everyone-running-the-huggingface-text-generation-inference-in-google-colab-5adb3218a137), which shows how to stream response.

```python
template = """<s>[INST] <<SYS>>
Provide a correct and short answer to the question.
<</SYS>>
{question} [/INST]"""

prompt = PromptTemplate(template = template, input_variables = ["question"])
```

```python
generation_config = dict(
    max_new_tokens = 64,
    top_k = 10,
    top_p = 0.8,
    temperature = 0.01,
    repetition_penalty = 1.03,
    streaming = True,
)

llm = HuggingFaceTextGenInference(inference_server_url = url, **generation_config)
chain = prompt | llm | StrOutputParser()
```

```python
chain.invoke({"question": "How do you make cheese ?"})
```

### 2.5 OpenAI client around TGI service

See the OpenAI-compatible [message API](https://huggingface.co/docs/text-generation-inference/messages_api), and this huggingface [blog post](https://huggingface.co/blog/tgi-messages-api).

```python
messages = [{
    'role': 'user',
    'content': '\n    Answer the question based on the context below. Keep your answer short. \n    Only use information mentioned in context to form your answer.\n    Respond "Unsure about answer" if not sure about the answer.\n\n    Context:\n    \nYour name is Jean-baptiste',
    },{
    'role': 'assistant', 
    'content': "Hi! How can I help you ?",
    },{
    'role': 'user', 
    'content': "What is the most beautiful city in the world ? please elaborate on your answer. Do not share your name.",
}]
```

```python
# init the client but point it to TGI
client = OpenAI(base_url = f"{url}/v1", api_key = "-")
```

```python
generation_config = dict(
    model = 'tgi',
    messages = messages,
    temperature = 0.1,
    top_p = .1,
    max_tokens = 64,
    stream = True,
    seed = 42,
)

chat_completion = client.chat.completions.create(**generation_config)
```

```python
for message in chat_completion:
    print(message.choices[0].delta.content, end = "")
```

```python

```

```python
# in vLLM:
# https://docs.mistral.ai/self-deployment/vllm/
# https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/discussions/29
# https://github.com/vllm-project/vllm/discussions/2112
```
