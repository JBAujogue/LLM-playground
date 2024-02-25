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
from typing import Dict, List

from datasets import load_dataset, DatasetDict
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

input_model_name = 'sentence-transformers/all-mpnet-base-v2'
output_model_name = 'all-mpnet-base-v2--18k-alpaca'

# input_model_name = 'thenlper/gte-base'
# output_model_name = 'gte-base--18k-alpaca'
```

```python
path_to_exp = os.path.join(path_to_runs, output_model_name)
```

```python
# this is for demo purpose only, should be set to None
max_samples = 500
```

### Custom package

```python
if path_to_src not in sys.path:
    sys.path.insert(0, path_to_src)
```

```python
from llmtools.trainers.sentence_transformers_trainers import EmbeddingQAFinetuneDataset, SentenceTransformersTrainer
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
# load dataset
dataset = load_dataset(dataset_name)
dataset = dataset['train']
if max_samples is not None:
    dataset = dataset.select(range(max_samples))
```

```python
for d in dataset.select(range(5)):
    print('QUERY:', d['instruction'])
    print('DOCUMENT:', d['output'])
    print('-----')
```

```python
# split into train / valid / test
dataset_train_else = dataset.train_test_split(test_size = .2, seed = 42, shuffle = False)
dataset_valid_test = dataset_train_else['test'].train_test_split(test_size = .5, seed = 42, shuffle = False)

# convert to dict of llama_index datasets
dataset_dict = dict(
    train = convert_hf_to_llamaindex_dataset(dataset_train_else['train']),
    valid = convert_hf_to_llamaindex_dataset(dataset_valid_test['train']),
    test  = convert_hf_to_llamaindex_dataset(dataset_valid_test['test']),
    all   = convert_hf_to_llamaindex_dataset(dataset),
)
```

# 2. Finetuning protocol selection

The llama-index `SentenceTransformersFinetuneEngine` [source code](https://github.com/run-llama/llama_index/blob/main/llama-index-finetuning/llama_index/finetuning/embeddings/sentence_transformer.py) is easily extractible into a standalone trainer class.

We considered using sentence-transformers's [callback](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/SentenceTransformer.py#L917) arg to write validation scores to a tensorboard log file, inspired by transformers tensorboard [callback](https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/integrations/integration_utils.py#L579) function. However the callback function is called over the evaluator's output, see [here](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/evaluation/InformationRetrievalEvaluator.py#L144), which is only a global score and not the fine-grained, per-metric scores.

```python
# model args and training args are listed in the sentence-transformers source code:
# https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/SentenceTransformer.py

model_args = dict(
    model_name_or_path = input_model_name,
    device = 'cpu',
    cache_folder = None,
)

training_args = dict(
    epochs = 1,
    steps_per_epoch = None,
    scheduler = 'warmupcosine',
    warmup_steps = 10,
    optimizer_params = {'lr': 2e-5},
    weight_decay = 1e-3,
    batch_size = 16,
    evaluation_steps = 5,
    save_best_model = False,
    max_grad_norm = 1,
    use_amp = False,
    callback = None,
    show_progress_bar = True,
    checkpoint_path = None,
    checkpoint_save_steps = 0,
    checkpoint_save_total_limit = 0,
)
```

```python
trainer = SentenceTransformersTrainer(
    model_args = model_args,
    logging_dir = path_to_exp,
    train_dataset = dataset_dict['train'],
    valid_dataset = dataset_dict['valid'],
    training_args = training_args,
)
```

### Evaluate model prior training

```python
trainer.evaluate(dataset_dict['test'], metric_key_prefix = 'base')
```

### Train model

```python
trainer.train()
```

### Evaluate model post training

```python
trainer.evaluate(dataset_dict['test'], metric_key_prefix = 'test')
```

# 3. Finetune final model

```python
trainer = SentenceTransformersTrainer(
    model_args = model_args,
    logging_dir = path_to_exp,
    train_dataset = dataset_dict['all'],
    valid_dataset = None,
    training_args = training_args | dict(evaluation_steps = 0),
)
```

### Train model

```python
trainer.train()
```

### Serialize trained model

```python
trainer.save_model()
```

```python

```
