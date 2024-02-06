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

References:
- Llama_index [documentation](https://docs.llamaindex.ai/en/stable/examples/finetuning/embeddings/finetune_embedding.html#run-embedding-finetuning)
- Example [blog post](https://betterprogramming.pub/fine-tuning-your-embedding-model-to-maximize-relevance-retrieval-in-rag-pipeline-2ea3fa231149)
- Huggingface OpenLLM [leaderboard](https://huggingface.co/spaces/mteb/leaderboard) for choosing base embedding model

### Packages

```python
%load_ext autoreload
%autoreload 2
```

```python
import os, sys
import re
from pathlib import Path

from datasets import load_dataset, load_from_disk, DatasetDict
from llama_index import Document
from llama_index.finetuning import EmbeddingQAFinetuneDataset, SentenceTransformersFinetuneEngine

from sentence_transformers import SentenceTransformer
```

```python
path_to_repo = Path(os.getcwd()).parent
path_to_data = os.path.join(path_to_repo, 'data', 'code-python')
path_to_logs = os.path.join(path_to_repo, 'experiments', 'logs')
path_to_models = os.path.join(path_to_repo, 'experiments', 'models')
```

### Global variables

```python
dataset_name = 'iamtarun--python_code_instructions_18k_alpaca'
input_model_name = 'sentence-transformers/all-mpnet-base-v2'
output_model_name = f'{input_model_name}@{dataset_name}'.replace('/', '--')
```

```python
path_to_exp = os.path.join(path_to_logs, output_model_name)

os.makedirs(path_to_exp, exist_ok = True)

run_list = [0] + [
    int(n)
    for f in os.listdir(path_to_exp) 
    if os.path.isdir(os.path.join(path_to_exp, f))
    for n in re.findall(r'^run(\d+)$', f)
]
path_to_run = os.path.join(path_to_exp, f'run{max(run_list)+1}')
```

# 1. Prepare dataset

```python
def convert_hf_to_llamaindex_dataset(dataset):
    return EmbeddingQAFinetuneDataset(
        queries = {i: t for i, t in enumerate(dataset['instruction'])},
        corpus = {i: t for i, t in enumerate(dataset['output'])}, 
        relevant_docs = {i: [i] for i in range(len(dataset))},
    )
```

```python
# load dataset from disk
dataset = load_from_disk(os.path.join(path_to_data, dataset_name))
dataset = dataset['train']

# split into train / valid / test
dataset_train_else = dataset.train_test_split(test_size = .2, seed = 42, shuffle = False)
dataset_valid_test = dataset_train_else['test'].train_test_split(test_size = .5, seed = 42, shuffle = False)

# convert to dict of llama_index datasets
dataset_dict = dict(
    train = convert_hf_to_llamaindex_dataset(dataset_train_else['train']),
    valid = convert_hf_to_llamaindex_dataset(dataset_valid_test['train']),
    test = convert_hf_to_llamaindex_dataset(dataset_valid_test['test']),
)
```

# 2. Finetune model

Remark: The llama-index `SentenceTransformersFinetuneEngine` [source code](https://github.com/run-llama/llama_index/blob/main/llama_index/finetuning/embeddings/sentence_transformer.py) is easily extractible into a standalone trainer class.

```python
training_config = dict(
    batch_size = 6,
    epochs = 1,
    evaluation_steps = 50,
)
```

```python
trainer = SentenceTransformersFinetuneEngine(
    dataset = dataset_dict['train'],
    val_dataset = dataset_dict['valid'],
    model_id = input_model_name,
    model_output_path = os.path.join(path_to_run, 'model'),
    **training_config,
)

trainer.finetune()
```

# 3. Evaluate model

```python
from torch.utils.tensorboard import SummaryWriter
```

```python
#writer = SummaryWriter(log_dir = )
```

```python
help(SummaryWriter)
```
