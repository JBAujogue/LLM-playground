"""
Sentence Transformer Trainer.
Adapted from
    https://github.com/run-llama/llama_index/blob/main/llama_index/finetuning/embeddings/sentence_transformer.py
    https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/integrations/integration_utils.py#L579
"""

from typing import Dict, Any, Optional
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
            if os.path.isdir(os.path.join(logging_dir, f))
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
        self.exportable_training_args = {
            k: v for k, v in training_args.items() 
            if v is None or isinstance(v, (int, float, list, tuple, dict, str, bool))
        }
        training_args |= dict(
            train_objectives = [(self.loader, self.loss)],
            evaluator = self.evaluator,
            output_path = self.output_path,
        )
        self.training_args = training_args

    def train(self):
        self.save_config()
        self.model.fit(**self.training_args)
        self.send_logs_to_tensorboard(metric_key_prefix = 'eval')
        return

    def evaluate(self, dataset: Any, metric_key_prefix: str = 'test', output_path: Optional[str] = None):
        evaluator = InformationRetrievalEvaluator(
            dataset.queries, dataset.corpus, dataset.relevant_docs,
        )
        out_path = (output_path or self.output_path)
        os.makedirs(out_path, exist_ok = True)
        score = evaluator(self.model, output_path = out_path)
        self.send_logs_to_tensorboard(metric_key_prefix, output_path = out_path)
        return score

    def save_config(self):
        os.makedirs(self.output_path, exist_ok = True)
        OmegaConf.save(self.exportable_training_args, os.path.join(self.output_path, 'config.yaml'))
        return

    def send_logs_to_tensorboard(self, metric_key_prefix: str, output_path: Optional[str] = None):
        log_dir = (output_path or self.output_path)
        self.tb_writer = self._SummaryWriter(log_dir = log_dir)
        
        # set path to logs 
        # TODO: try to infer it from self attributes
        log_filepath = os.path.join(
            log_dir, 
            (metric_key_prefix if metric_key_prefix == 'eval' else ''), 
            'Information-Retrieval_evaluation_results.csv',
        )
        if os.path.isfile(log_filepath):
            # parse sentence-transformers scores
            tb_scores = pd.read_csv(log_filepath).to_dict('records')
    
            # write scores to tensorboard log file
            max_steps = max(scores['steps'] for scores in tb_scores)
            for scores in tb_scores:
                epoch, step = scores['epoch'], scores['steps']
                if step != -1:
                    for k, v in scores.items():
                        if isinstance(v, (int, float)):
                            self.tb_writer.add_scalar(f'{metric_key_prefix}/{k}', v, epoch * max_steps + step)
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