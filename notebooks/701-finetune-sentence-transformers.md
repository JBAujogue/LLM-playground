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
from llama_index.finetuning import EmbeddingQAFinetuneDataset
```

```python
path_to_repo = Path(os.getcwd()).parent
path_to_data = os.path.join(path_to_repo, 'data')
path_to_runs = os.path.join(path_to_repo, 'mlruns')
```

### Global variables

```python
# input
dataset_folder = 'code-python'
dataset_name = 'iamtarun--python_code_instructions_18k_alpaca'
input_model_name = 'sentence-transformers/all-mpnet-base-v2'
```

```python
# output
output_model_name = f'{input_model_name}@{dataset_folder}'.replace('/', '--')

path_to_exp = os.path.join(path_to_runs, output_model_name)
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
dataset = load_from_disk(os.path.join(path_to_data, dataset_folder, dataset_name))
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

The llama-index `SentenceTransformersFinetuneEngine` [source code](https://github.com/run-llama/llama_index/blob/main/llama_index/finetuning/embeddings/sentence_transformer.py) is easily extractible into a standalone trainer class:

```python
# from llama_index.finetuning import SentenceTransformersFinetuneEngine
```

```python
"""
Sentence Transformer Trainer.
Adapted from
    https://github.com/run-llama/llama_index/blob/main/llama_index/finetuning/embeddings/sentence_transformer.py
"""

from typing import Dict, Any, Optional


class SentenceTransformersTrainer:
    """
    Sentence Transformers Trainer.
    """
    def __init__(
        self,
        model_name_or_path: str,
        logging_dir: str,
        train_dataset: Any,
        valid_dataset: Optional[Any] = None,
        batch_size: int = 8,
        training_args: Optional[Dict[str, Any]] = {},
        ):
        """
        Init params.
        training_args are listed in 
            https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/SentenceTransformer.py
        """
        import os
        import re
        from torch.utils.data import DataLoader
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from sentence_transformers.evaluation import InformationRetrievalEvaluator

        # set output path
        os.makedirs(logging_dir, exist_ok = True)
        run_list = [0] + [
            int(n)
            for f in os.listdir(logging_dir) 
            if os.path.isdir(os.path.join(path_to_exp, f))
            for n in re.findall(r'^run(\d+)$', f)
        ]
        self.output_path = os.path.join(logging_dir, f'run{max(run_list)+1}')

        # set model
        self.model = SentenceTransformer(model_name_or_path, device = 'cuda')

        # set dataloader
        self.examples = [
            InputExample(texts = [query, train_dataset.corpus[text_id]])
            for query_id, query in train_dataset.queries.items()
            for text_id in train_dataset.relevant_docs[query_id]
        ]
        self.loader = DataLoader(self.examples, batch_size = batch_size)

        # set loss
        self.loss = losses.MultipleNegativesRankingLoss(self.model)
        
        # set evaluator
        evaluator: Optional[InformationRetrievalEvaluator] = None
        if valid_dataset is not None:
            evaluator = InformationRetrievalEvaluator(
                valid_dataset.queries, valid_dataset.corpus, valid_dataset.relevant_docs,
            )
        self.evaluator = evaluator

        # set default training arguments
        training_args |= dict(
            train_objectives = [(self.loader, self.loss)],
            evaluator = self.evaluator,
            output_path = self.output_path,
        )
        self.training_args = training_args

    def train(self):
        self.model.fit(**self.training_args)

    def evaluate(self, dataset: Any, output_path: str):
        evaluator = InformationRetrievalEvaluator(
            dataset.queries, dataset.corpus, dataset.relevant_docs,
        )
        return evaluator(self.model, output_path = output_path)
```

We may use sentence-transformers's [callback](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/SentenceTransformer.py#L917) arg to write validation scores to a tensorboard log file, see pytorch's [Tensorboard](https://pytorch.org/docs/stable/tensorboard.html) doc

```python
# from torch.utils.tensorboard import SummaryWriter
```

```python
training_args = dict(
    epochs = 2,
    steps_per_epoch = None,
    scheduler = "WarmupLinear",
    warmup_steps = 100,
    optimizer_params = {"lr": 2e-5},
    weight_decay = 1e-3,
    evaluation_steps = 200,
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
    model_name_or_path = input_model_name,
    logging_dir = path_to_exp,
    train_dataset = dataset_dict['train'],
    valid_dataset = dataset_dict['valid'],
    batch_size = 6,
    training_args = training_args,
)
```

```python
trainer.train()
```

# 3. Evaluate model



```python

```

```python
out_path = trainer.training_args['output_path']
out_path
```
