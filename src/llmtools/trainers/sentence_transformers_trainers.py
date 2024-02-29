"""
Sentence Transformer Trainer.
Adapted from
    https://github.com/run-llama/llama_index/blob/main/llama_index/finetuning/embeddings/sentence_transformer.py
    https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/integrations/integration_utils.py#L579
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

import os
import re
import pandas as pd

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator


logger = logging.getLogger(__name__)


@dataclass
class EmbeddingQAFinetuneDataset:
    '''
    Embedding QA Finetuning Dataset.

    Args:
        queries (Dict[str, str]): Dict id -> query.
        corpus (Dict[str, str]): Dict id -> string.
        relevant_docs (Dict[str, List[str]]): Dict query id -> list of doc ids.
    '''
    queries: Dict[str, str]              # dict id -> query
    corpus: Dict[str, str]               # dict id -> string
    relevant_docs: Dict[str, List[str]]  # query id -> list of doc ids
    mode: str = "text"


class SentenceTransformersTrainer:
    '''
    Simple Trainer for sentence-transformers models.
    '''
    def __init__(
        self,
        model_args: Dict[str, Any],
        logging_dir: str,
        train_dataset: Any,
        valid_dataset: Optional[Any] = None,
        training_args: Optional[Dict[str, Any]] = {},
        ):
        '''
        Init params.
        training_args are listed in 
            https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/SentenceTransformer.py
        '''
        # set output path
        self.run_dir = self.compute_run_dir(logging_dir)

        # set training args
        self.training_args = dict(training_args)
        self.exportable_training_args = {
            k: v for k, v in self.training_args.items() 
            if v is None or isinstance(v, (int, float, list, tuple, dict, str, bool))
        }
        if 'batch_size' in self.training_args:
            batch_size = self.training_args['batch_size']
            self.training_args.pop('batch_size')
        else:
            batch_size = 8

        # set model
        self.model = SentenceTransformer(**model_args)

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
            self.training_args |= dict(output_path = self.run_dir)
            evaluator = InformationRetrievalEvaluator(
                valid_dataset.queries, valid_dataset.corpus, valid_dataset.relevant_docs,
            )
        self.evaluator = evaluator

        # set tensorboard log writer
        self._SummaryWriter = SummaryWriter

        # set default training arguments
        self.training_args |= dict(
            train_objectives = [(self.loader, self.loss)],
            evaluator = self.evaluator,
        )

    def compute_run_dir(self, logging_dir):
        '''
        creates subfolder 'runXX' within input folder 'logging_dir',
        where XX is the smallest integer number possible.
        '''
        os.makedirs(logging_dir, exist_ok = True)
        run_list = [0] + [
            int(n)
            for f in os.listdir(logging_dir) 
            if os.path.isdir(os.path.join(logging_dir, f))
            for n in re.findall(r'^run(\d+)$', f)
        ]
        return os.path.join(logging_dir, f'run{max(run_list)+1}')

    def train(self):
        '''
        Train model according to training args specified in intanciation.
        Log results into a tensorboard summary.
        '''
        logger.info(
            f'''
            Using the following finetuning config:
            {self.exportable_training_args}
            '''
        )
        self.save_config()
        self.model.fit(**self.training_args)
        if self.evaluator is not None:
            self.send_logs_to_tensorboard(metric_key_prefix = 'eval')
        return

    def evaluate(self, dataset: Any, metric_key_prefix: str = 'test'):
        '''
        Evaluate model on an input dataset.
        Log results into a tensorboard summary.
        '''
        evaluator = InformationRetrievalEvaluator(
            dataset.queries, dataset.corpus, dataset.relevant_docs,
        )
        out_path = os.path.join(self.run_dir, metric_key_prefix)
        os.makedirs(out_path, exist_ok = True)
        score = evaluator(self.model, output_path = out_path)
        self.send_logs_to_tensorboard(metric_key_prefix)
        return score

    def save_config(self):
        '''
        Save training config into a yaml file.
        '''
        os.makedirs(self.run_dir, exist_ok = True)
        out_path = os.path.join(self.run_dir, 'config.yaml')
        OmegaConf.save(self.exportable_training_args, out_path)
        return

    def save_model(self, model_path: Optional[str] = None):
        '''
        Save model into the specified folder path.
        '''
        out_path = model_path or os.path.join(self.run_dir, 'model')
        self.model.save(out_path)
        return out_path

    def send_logs_to_tensorboard(self, metric_key_prefix: str):
        '''
        Maps the evaluation report .csv file generated by
        sentence-transformers into a tensorboard summary.
        '''
        tb_writer = self._SummaryWriter(log_dir = self.run_dir)
        
        # set path to logs 
        out_path = os.path.join(
            self.run_dir, 
            metric_key_prefix, 
            'Information-Retrieval_evaluation_results.csv',
        )
        if os.path.isfile(out_path):
            # parse sentence-transformers scores
            tb_scores = pd.read_csv(out_path).to_dict('records')
    
            # write scores to tensorboard log file
            max_epoch = max(scores['epoch'] for scores in tb_scores)
            max_steps = max(scores['steps'] for scores in tb_scores) + 1
            for scores in tb_scores:
                epoch, step = scores['epoch'], scores['steps']
                if max_epoch == -1:
                    epoch = 0
                if step == -1:
                    step = max_steps
    
                for k, v in scores.items():
                    if isinstance(v, (int, float)):
                        tb_writer.add_scalar(f'{metric_key_prefix}/{k}', v, epoch * max_steps + step)
                    else:
                        logger.warning(
                            "Trainer is attempting to log a value of "
                            f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                            "This invocation of Tensorboard's writer.add_scalar() "
                            "is incorrect so we dropped this attribute."
                        )
                tb_writer.flush()
        tb_writer.close()
        return
