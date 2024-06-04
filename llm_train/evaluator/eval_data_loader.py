import atexit
import os
import pickle
import sys
import traceback

import torch
from datasets import load_dataset

from transformers import pipeline

import constant_config

disease_ner_pipeline = None

datasets_dict = None


def resolve(filename):
    """
    Search file in current system paths
    :param filename:
    :return:
    """
    for directory in sys.path:
        path = os.path.join(directory, filename)
        if os.path.isfile(path):
            return path


def load_eval_data_dict():
    """
    Load in memory (datasets_dict) the evaluation datasets from the evaluation dataset pickle object file
    stored at constant_config.EVAL_DATA_PICKLE_FILE_PATH
    """
    global datasets_dict

    if datasets_dict is not None:
        return

    # Load datasets_dict
    if os.path.isfile(constant_config.EVAL_DATA_PICKLE_FILE_PATH):
        print(f"Loading evaluation data (datasets_dict) from pickle file {constant_config.EVAL_DATA_PICKLE_FILE_PATH}...")
        file = open(constant_config.EVAL_DATA_PICKLE_FILE_PATH, 'rb')
        datasets_dict = pickle.load(file)
        file.close()
    else:
        print(f"Impossible to load evaluation data (datasets_dict) from pickle file {constant_config.EVAL_DATA_PICKLE_FILE_PATH}...")
        datasets_dict = dict()

    atexit.register(dump_eval_data_dict)


def dump_eval_data_dict():
    """
    Store the evaluation datasets (datasets_dict) as the evaluation dataset pickle object file
    stored at constant_config.EVAL_DATA_PICKLE_FILE_PATH
    """
    global datasets_dict

    # Save datasets_dict
    print(f"Storing evaluation data (datasets_dict) to pickle file {constant_config.EVAL_DATA_PICKLE_FILE_PATH}...")
    file = open(constant_config.EVAL_DATA_PICKLE_FILE_PATH, 'wb')
    pickle.dump(datasets_dict, file)
    file.close()


def init_disease_ner():
    """
    Init HF disease NER pipeline --> https://huggingface.co/pruas/BENT-PubMedBERT-NER-Disease
    """
    global disease_ner_pipeline
    if disease_ner_pipeline is None:
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        print(f"Loading disease NER pipeline on {device}...")
        disease_ner_pipeline = pipeline("token-classification", model="pruas/BENT-PubMedBERT-NER-Disease", device=device)


def disease_ner(text):
    """
    Return disease mentions contained in the text, if ay
    :param text: input text where to spot disease mentions
    :return: list of disease mentions
    """
    global disease_ner_pipeline
    init_disease_ner()

    disease_mentions = None
    try:
        disease_mentions = disease_ner_pipeline(text)
    except Exception:
        traceback.print_stack()

    return disease_mentions


def load_biosses(generate_disease_only=False):
    """
    Load BIOSSES sentence pairs with similarity score - see https://huggingface.co/datasets/biosses
    :param generate_disease_only: (default False) if set to True, also the biosses__disease dataset will be loaded
    :return: 2 datasets, biosses__all and biosses__disease
    Note that the biosses__disease datasets includes only sentences pairs where both sentences contain one or more
    mentions of diseases
    """
    load_eval_data_dict()
    global datasets_dict

    if 'biosses__all' not in datasets_dict:
        datasets_dict['biosses__all'] = None
    if 'biosses__disease' not in datasets_dict:
        datasets_dict['biosses__disease'] = None

    # Load the BIOSSES dataset, if not already done
    if datasets_dict['biosses__all'] is None:
        print(f"Loading the BIOSSES dataset (all)...")

        dataset = load_dataset("biosses")

        datasets_dict['biosses__all'] = dataset['train']

        dump_eval_data_dict()

    if datasets_dict['biosses__all'] is not None:
        print(f"Loaded BIOSSES dataset: biosses__all {len(datasets_dict['biosses__all'])} sentence pairs.")

    # Load the BIOSSES dataset (only diseases), if not already done
    if generate_disease_only is True and datasets_dict['biosses__disease'] is None:
        print(f"Loading the BIOSSES dataset (only diseases)...")

        def is_disease_sample(example):
            sent_1 = disease_ner(example['sentence1'])
            sent_2 = disease_ner(example['sentence2'])

            if sent_1 is not None and len(sent_1) > 0 and sent_2 is not None and len(sent_2) > 0:
                return True
            else:
                return False

        datasets_dict['biosses__disease'] = datasets_dict['biosses__all'].filter(is_disease_sample)

        dump_eval_data_dict()

    if datasets_dict['biosses__disease'] is not None:
        print(f"Loaded BIOSSES dataset: biosses__disease {len(datasets_dict['biosses__disease'])} sentence pairs.")

    return datasets_dict['biosses__all'], datasets_dict['biosses__disease']


def load_semeval_sts_test_mteb(generate_disease_only=False):
    """
    Load Semeval STS test sets (English) similarity score - see https://huggingface.co/datasets/mteb/stsTT-sts
    where TT = ["12", "13", "14", "15", "16"]
    :param generate_disease_only: (default False) if set to True, also the sts__disease datasets will be loaded
    :return: 5 datasets, "STS_12", "STS_13", "STS_14", "STS_15", "STS_16"
    Note that the sts__disease datasets includes only sentences pairs where both sentences contain one or more
    mentions of diseases
    """
    load_eval_data_dict()
    global datasets_dict

    for TT in ["12", "13", "14", "15", "16"]:

        if f'STS_{TT}__all' not in datasets_dict:
            datasets_dict[f'STS_{TT}__all'] = None
        if f'STS_{TT}__disease' not in datasets_dict:
            datasets_dict[f'STS_{TT}__disease'] = None

        # Load the SemEval STS dataset (all), if not already done
        if datasets_dict[f'STS_{TT}__all'] is None:
            print(f"Loading the STS_{TT} dataset (all)...")
            dataset = load_dataset(f"mteb/sts{TT}-sts")
            datasets_dict[f'STS_{TT}__all'] = dataset['test']

            dump_eval_data_dict()

        if datasets_dict[f'STS_{TT}__all'] is not None:
            print(f"Loaded SemEval {TT} STS (English) dataset - STS_{TT}__all {len(datasets_dict[f'STS_{TT}__all'])} sentence pairs.")

        # Load the SemEval STS dataset (only diseases), if not already done
        if generate_disease_only is True and datasets_dict[f'STS_{TT}__disease'] is None:
            print(f"Loading the STS_{TT} dataset (only disease)...")

            def is_disease_sample(example):
                sent_1 = disease_ner(example['sentence1'])
                sent_2 = disease_ner(example['sentence2'])

                if sent_1 is not None and len(sent_1) > 0 and sent_2 is not None and len(sent_2) > 0:
                    return True
                else:
                    return False

            datasets_dict[f'STS_{TT}__disease'] = datasets_dict[f'STS_{TT}__all'].filter(is_disease_sample)

            dump_eval_data_dict()

        if datasets_dict[f'STS_{TT}__disease'] is not None:
            print(f"Loaded SemEval {TT} STS (English) dataset - STS_{TT}__disease {len(datasets_dict[f'STS_{TT}__disease'])} sentence pairs.")

    return {f'STS_{TT}__all': datasets_dict[f'STS_{TT}__all'] for TT in ["12", "13", "14", "15", "16"]}, {f'STS_{TT}__disease': datasets_dict[f'STS_{TT}__disease'] for TT in ["12", "13", "14", "15", "16"]}


if __name__ == '__main__':

    # Load BIOSSES
    biosses__all, biosses__disease = load_biosses(generate_disease_only=True)

    # Load SemEval STS EN
    semeval_sts_dict__all, semeval_sts_dict__disease = load_semeval_sts_test_mteb(generate_disease_only=True)
