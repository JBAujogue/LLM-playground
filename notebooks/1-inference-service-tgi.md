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

# Backend LLM service using Text Generation Inference (TGI)

### Packages

```python
%load_ext autoreload
%autoreload 2
```

```python
import os, sys
import time
import requests
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

### Start backend service

<!-- #region -->
Run the backend service with one of the following methods:

1. Create and start a new container: open a CLI at the root of the project, and run the command
```bash
wsl -e ./scripts/services/tgi-service-creation.sh
```
2. Start an existing container in CLI: open a CLI at the root of the project, and run the command
```bash
wsl -e ./scripts/services/tgi-service.sh
```
3. Start an existing container from Docker Desktop: Open the Docker Desktop client and run the appropriate container.
4. Start an existing container with Docker's python API:
```python
container_name = 'tgi-service'
container = docker.from_env().containers.get(container_name)
container.start()
```
see [docker-py](https://docker-py.readthedocs.io/en/stable/containers.html) for details.
<!-- #endregion -->

```python
url = 'http://127.0.0.1:8080'
```

***


### 1. Simple client around TGI service

See the TGI [docs](https://huggingface.co/docs/text-generation-inference/quicktour).

```python
template = """<s>[INST] <<SYS>> {system} <</SYS>> {question} [/INST]"""
system = "Provide a correct and short answer to the question."
question = "How do you make cheese ?"

query = template.replace('{system}', system).replace('{question}', question)
```

```python
headers = {"Content-Type": "application/json"}

generation_config = dict(
    do_sample = True,
    temperature = 0.01,
    top_k = 10,
    top_p = 0.8,
    max_new_tokens = 64,
    repetition_penalty = 1.03,
    stream = True,
    seed = 42,
)

data = {
    'inputs': query,
    'parameters': generation_config,
}
response = requests.post(f'{url}/generate', headers = headers, json = data)
```

```python
response.json()['generated_text']
```

### 2. Text-generation client around TGI service

**Not working on local endpoints**. See [Huggingface's official python client](https://github.com/huggingface/text-generation-inference/tree/main/clients/python) around TGI services.

```python
# client = InferenceAPIClient(url)
# client = InferenceAPIAsyncClient(url)
```

### 3. Huggingface-hub client around TGI service

See [Huggingface-hub python client](https://huggingface.co/docs/text-generation-inference/basic_tutorials/consuming_tgi) around TGI services.

```python
template = """<s>[INST] <<SYS>> {system} <</SYS>> {question} [/INST]"""
system = "Provide a correct and short answer to the question."
question = "How do you make cheese ?"

query = template.replace('{system}', system).replace('{question}', question)
```

```python
client = InferenceClient(model = url)
```

```python
generation_config = dict(
    do_sample = True,
    temperature = 0.01,
    top_k = 10,
    top_p = 0.8,
    max_new_tokens = 64,
    repetition_penalty = 1.03,
    stream = True,
    seed = 42,
)

response = client.text_generation(query, **generation_config)
```

```python
for token in response:
    print(token, end = '', flush = True)
```

### 4. Langchain client around TGI service

See the [langchain community official doc](https://python.langchain.com/docs/integrations/llms/huggingface_endpoint) (also see this [deprecated page](https://api.python.langchain.com/en/latest/llms/langchain_community.llms.huggingface_text_gen_inference.HuggingFaceTextGenInference.html#langchain-community-llms-huggingface-text-gen-inference-huggingfacetextgeninference) in case of trouble), and [this example](https://python.langchain.com/docs/expression_language/streaming#chains), which shows how to stream response.

```python
template = """<s>[INST] <<SYS>> {system} <</SYS>> {question} [/INST]"""
system = "Provide a correct and short answer to the question."
question = "How do you make cheese ?"

prompt = PromptTemplate(template = template, input_variables = ["system", "question"])
```

```python
generation_config = dict(
    do_sample = True,
    temperature = 0.01,
    top_k = 10,
    top_p = 0.8,
    max_new_tokens = 64,
    repetition_penalty = 1.03,
    streaming = True,
    seed = 42,
)

llm = HuggingFaceTextGenInference(inference_server_url = url, **generation_config)
chain = prompt | llm | StrOutputParser()
```

```python
response = chain.astream(dict(system = system, question = question))

async for chunk in response:
    print(chunk, end = '', flush = True)
```

### 5. OpenAI client around TGI service

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
    print(message.choices[0].delta.content, end = '', flush = True)
```

```python

```
