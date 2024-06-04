from sentence_transformers import SentenceTransformer, InputExample
import logging

from sentence_transformers.evaluation import SimilarityFunction, EmbeddingSimilarityEvaluator
from typing import List, Literal, Optional


logger = logging.getLogger(__name__)


class CustomEmbeddingSimilarityEvaluator(EmbeddingSimilarityEvaluator):

    def __init__(
        self,
        sentences1: List[str],
        sentences2: List[str],
        scores: List[float],
        batch_size: int = 16,
        main_similarity: SimilarityFunction = None,
        name: str = "",
        show_progress_bar: bool = False,
        write_csv: bool = True,
        precision: Optional[Literal["float32", "int8", "uint8", "binary", "ubinary"]] = None,
        truncate_dim: Optional[int] = None,
    ):
        super().__init__(sentences1, sentences2, scores, batch_size, main_similarity, name, show_progress_bar, write_csv, precision, truncate_dim)

        self.detailed_eval_results = dict()

    def __call__(self, model: SentenceTransformer, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:

        metrics = super().__call__(model, output_path, epoch, steps)

        self.detailed_eval_results = dict()
        self.detailed_eval_results['name'] = self.name
        self.detailed_eval_results['model'] = f'{model}'
        self.detailed_eval_results['epoch'] = epoch
        self.detailed_eval_results['steps'] = steps
        self.detailed_eval_results['eval_spearman_cosine'] = metrics[f'{self.name}_spearman_cosine']
        self.detailed_eval_results['eval_pearson_cosine'] = metrics[f'{self.name}_pearson_cosine']
        self.detailed_eval_results['eval_spearman_euclidean'] = metrics[f'{self.name}_spearman_euclidean']
        self.detailed_eval_results['eval_pearson_euclidean'] = metrics[f'{self.name}_pearson_euclidean']
        self.detailed_eval_results['eval_spearman_manhattan'] = metrics[f'{self.name}_spearman_manhattan']
        self.detailed_eval_results['eval_pearson_manhattan'] = metrics[f'{self.name}_pearson_manhattan']
        self.detailed_eval_results['eval_spearman_dot'] = metrics[f'{self.name}_spearman_dot']
        self.detailed_eval_results['eval_pearson_dot'] = metrics[f'{self.name}_pearson_dot']

        if self.main_similarity == SimilarityFunction.COSINE:
            return metrics[f'{self.name}_spearman_cosine']
        elif self.main_similarity == SimilarityFunction.EUCLIDEAN:
            return metrics[f'{self.name}_spearman_euclidean']
        elif self.main_similarity == SimilarityFunction.MANHATTAN:
            return metrics[f'{self.name}_spearman_manhattan']
        elif self.main_similarity == SimilarityFunction.DOT_PRODUCT:
            return metrics[f'{self.name}_spearman_dot']
        elif self.main_similarity is None:
            return max(metrics[f'{self.name}_spearman_cosine'], metrics[f'{self.name}_spearman_manhattan'], metrics[f'{self.name}_spearman_euclidean'], metrics[f'{self.name}_spearman_dot'])
        else:
            raise ValueError("Unknown main_similarity value")
