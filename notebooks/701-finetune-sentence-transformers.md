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
from datasets import Dataset

dataset = Dataset.from_dict({
    "instruction": [
        "write me a message", 
        "give me a cover letter",
        "write me a love letter",
        'lol',
        'give me a joke',
        'I like cakes',
        'bigO',
        'some text',
        "write me a message", 
        "give me a cover letter",
        "write me a love letter",
        'lol',
        'give me a joke',
        'I like cakes',
        'bigO',
        'some text',
    ], 
    "output": [
        'some answer', 
        'some funny story',
        'hey',
        "cover letter",
        "some message",
        "love letter",
        'lol',
        'some cake recipe',
        'some answer', 
        'some funny story',
        'hey',
        "cover letter",
        "some message",
        "love letter",
        'lol',
        'some cake recipe',
    ]})

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

The llama-index `SentenceTransformersFinetuneEngine` [source code](https://github.com/run-llama/llama_index/blob/main/llama_index/finetuning/embeddings/sentence_transformer.py) is easily extractible into a standalone trainer class.

We considered using sentence-transformers's [callback](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/SentenceTransformer.py#L917) arg to write validation scores to a tensorboard log file, inspired by transformers tensorboard [callback](https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/integrations/integration_utils.py#L579) function. However the callback function is called over the evaluator's output, see [here](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/evaluation/InformationRetrievalEvaluator.py#L144), which is only a global score and not the fine-grained, per-metric scores.

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
import logging

import os
import re
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator


logger = logging.getLogger(__name__)


class SentenceTransformersTrainer:
    """
    Sentence Transformers Trainer.
    """
    def __init__(
        self,
        model_name_or_path: str,
        device: str,
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
        self.model = SentenceTransformer(model_name_or_path, device = device)

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

        # set tensorboard summary writer
        self._SummaryWriter = SummaryWriter

        # set default training arguments
        training_args |= dict(
            train_objectives = [(self.loader, self.loss)],
            evaluator = self.evaluator,
            output_path = self.output_path,
        )
        self.training_args = training_args

    def train(self):
        self.model.fit(**self.training_args)
        self.send_logs_to_tensorboard('eval')
        return

    def evaluate(self, dataset: Any, metric_key_prefix: str = 'test', output_path: Optional[str] = None):
        evaluator = InformationRetrievalEvaluator(
            dataset.queries, dataset.corpus, dataset.relevant_docs,
        )
        out_path = (output_path or self.output_path)
        score = evaluator(self.model, output_path = out_path)
        self.send_logs_to_tensorboard(metric_key_prefix, output_path = out_path)
        return score

    def send_logs_to_tensorboard(self, metric_key_prefix: str, output_path: Optional[str] = None):
        import pandas as pd

        self.tb_writer = self._SummaryWriter(log_dir = self.output_path)
        
        # set path to logs 
        # TODO: try to infer it from self attributes
        logs_filepath = os.path.join(
            (output_path or self.output_path), 
            (metric_key_prefix if metric_key_prefix == 'eval' else ''), 
            'Information-Retrieval_evaluation_results.csv',
        )
        if os.path.isfile(logs_filepath):
            # parse sentence-transformers scores
            tb_scores = pd.read_csv(logs_filepath).to_dict('records')
    
            # write scores to tensorboard log file
            for scores in tb_scores:
                epoch, step = scores['epoch'], scores['steps']
                for k, v in scores.items():
                    if isinstance(v, (int, float)):
                        self.tb_writer.add_scalar(f'{metric_key_prefix}/{k}', v, (epoch + 1) * step)
                    else:
                        logger.warning(
                            "Trainer is attempting to log a value of "
                            f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                            "This invocation of Tensorboard's writer.add_scalar() "
                            "is incorrect so we dropped this attribute."
                        )
                self.tb_writer.flush()
        self.tb_writer.close()
        return
```

```python
training_args = dict(
    epochs = 2,
    steps_per_epoch = None,
    scheduler = "WarmupLinear",
    warmup_steps = 100,
    optimizer_params = {"lr": 2e-5},
    weight_decay = 1e-3,
    evaluation_steps = 2,
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
    device = 'cpu',
    logging_dir = path_to_exp,
    train_dataset = dataset_dict['train'],
    valid_dataset = dataset_dict['valid'],
    batch_size = 1,
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
trainer.evaluate(dataset_dict['test'])
```
