import copy
import json

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SimilarityFunction

from llm_train.evaluator.custom_evaluators import CustomEmbeddingSimilarityEvaluator
from llm_train.evaluator.eval_data_loader import load_semeval_sts_test_mteb, load_biosses


class EVALUATORS_STS:
    BIOSSES_ALL = "BIOSSES__all"
    BIOSSES_DISEASE = "BIOSSES__disease"
    SEMEVAL_STS_12_ALL = "STS_12__all"
    SEMEVAL_STS_12_DISEASE = "STS_12__disease"
    SEMEVAL_STS_13_ALL = "STS_13__all"
    SEMEVAL_STS_13_DISEASE = "STS_13__disease"
    SEMEVAL_STS_14_ALL = "STS_14__all"
    SEMEVAL_STS_14_DISEASE = "STS_14__disease"
    SEMEVAL_STS_15_ALL = "STS_15__all"
    SEMEVAL_STS_15_DISEASE = "STS_15__disease"
    SEMEVAL_STS_16_ALL = "STS_16__all"
    SEMEVAL_STS_16_DISEASE = "STS_16__disease"


def get_STS_evaluator(eval_id, batch_size=16, sim_score=SimilarityFunction.COSINE):
    """
    Instantiate and return a CustomEmbeddingSimilarityEvaluator (llm_train.evaluator.custom_evaluators) instance
    able to evaluate against the dataset specified by eval_id, with the provided embedding-LLM batch size and
    similarity score function
    :param eval_id: one of EVALUATORS_STS
    :param batch_size: batch size to embed textual contents through the provided embedding-LLM
    :param sim_score: similarity score function to use
    :return: CustomEmbeddingSimilarityEvaluator (llm_train.evaluator.custom_evaluators) instance
    """
    sent_list_1 = list()
    sent_list_2 = list()
    scores = list()
    name = "UNDEFINED"

    if eval_id == EVALUATORS_STS.BIOSSES_ALL:
        # Similarity score: float ranging from 0 (no relation) to 4 (equivalent)
        biosses__all,  _ = load_biosses(generate_disease_only=True)
        sent_list_1 = biosses__all["sentence1"]
        sent_list_2 = biosses__all["sentence2"]
        scores = biosses__all["score"]
        name = "BIOSSES__all"
    elif eval_id == EVALUATORS_STS.BIOSSES_DISEASE:
        # Similarity score: float ranging from 0 (no relation) to 4 (equivalent)
        _, biosses__disease = load_biosses(generate_disease_only=True)
        sent_list_1 = biosses__disease["sentence1"]
        sent_list_2 = biosses__disease["sentence2"]
        scores = biosses__disease["score"]
        name = "BIOSSES__disease"
    elif eval_id in [EVALUATORS_STS.SEMEVAL_STS_12_ALL, EVALUATORS_STS.SEMEVAL_STS_13_ALL, EVALUATORS_STS.SEMEVAL_STS_14_ALL, EVALUATORS_STS.SEMEVAL_STS_15_ALL, EVALUATORS_STS.SEMEVAL_STS_16_ALL,
                     EVALUATORS_STS.SEMEVAL_STS_12_DISEASE, EVALUATORS_STS.SEMEVAL_STS_13_DISEASE, EVALUATORS_STS.SEMEVAL_STS_14_DISEASE, EVALUATORS_STS.SEMEVAL_STS_15_DISEASE, EVALUATORS_STS.SEMEVAL_STS_16_DISEASE]:
        # Similarity score: float ranging from 0 (no relation) to 5 (equivalent) where:
        # (5) The two sentences are completely equivalent, as they mean the same thing.
        # (4) The two sentences are mostly equivalent, but some unimportant details differ.
        # (3) The two sentences are roughly equivalent, but some important information differs/missing.
        # (2) The two sentences are not equivalent, but share some details.
        # (1) The two sentences are not equivalent, but are on the same topic.
        # (0) The two sentences are on different topics.
        semeval_sts_dict__all, semeval_sts_dict__disease = load_semeval_sts_test_mteb(generate_disease_only=True)
        semeval_sts_dict = semeval_sts_dict__all if "all" in eval_id else semeval_sts_dict__disease

        sent_list_1 = semeval_sts_dict[eval_id]["sentence1"]
        sent_list_2 = semeval_sts_dict[eval_id]["sentence2"]
        scores = semeval_sts_dict[eval_id]["score"]
        name = eval_id
    else:
        print(f"Impossible to load STS evaluator with ID: {eval_id}")
        return None

    print(f"Loading STS evaluator with ID: {eval_id}")

    return CustomEmbeddingSimilarityEvaluator(sent_list_1, sent_list_2,
                                              scores,
                                              batch_size=batch_size,
                                              main_similarity=sim_score,
                                              name=name,
                                              show_progress_bar=True,
                                              write_csv=True,
                                              precision=None,
                                              truncate_dim=None)


def evaluate_sts_datasets(sentence_trs_model, epoch=-1, steps=-1, batch_size=128):
    """
    Perform the evaluation of the sentence transformer model provided as parameter against the sentence similarity
    datasets, that are BIOASSES and SemEval Sentence Similarity from 2012 to 2016
    :param sentence_trs_model: sentence transformer model to evaluate
    :param epoch: epoch number at which the evaluation is performed
    :param steps: step number at which the evaluation is performed
    :param batch_size: batch size to use during evaluation to generate text embedding through the considered sentence
                       transformer (i.e. embedding-LLM)
    :return: dictionary with keys equal to the evaluation identifier (i.e. EVALUATORS_STS) and values equal to
             the evaluation scores / results (i.e. Spearman correlation)
    """
    all_sts_evaluators_ret_dict = dict()

    # Evaluate the sentence_trs_model against all the EVALUATORS
    for evaluator_name, evaluator_identifier in vars(EVALUATORS_STS).items():
        if isinstance(evaluator_identifier, str) and not evaluator_name.startswith("__"):
            evaluator_instance = get_STS_evaluator(eval_id=evaluator_identifier, batch_size=batch_size)
            evaluator_score = evaluator_instance(sentence_trs_model, epoch=epoch, steps=steps)
            print(f"{evaluator_identifier} --> {evaluator_score}")
            all_sts_evaluators_ret_dict[evaluator_identifier] = copy.deepcopy(evaluator_instance.detailed_eval_results)

    return all_sts_evaluators_ret_dict


if __name__ == '__main__':
    # Load BIOSSES
    biosses__all, biosses__disease = load_biosses(generate_disease_only=True)

    # Load SemEval STS EN
    semeval_sts_list__all, semeval_sts_list__disease = load_semeval_sts_test_mteb(generate_disease_only=True)

    # Evaluation call
    sentence_transformer_model = SentenceTransformer("avsolatorio/GIST-Embedding-v0", trust_remote_code=True)
    all_sts_evaluators_dict = evaluate_sts_datasets(sentence_transformer_model, epoch=1, steps=1, batch_size=128)
    print(f"Loaded all STS evaluators: {json.dumps(all_sts_evaluators_dict, indent=4, sort_keys=True)}")
