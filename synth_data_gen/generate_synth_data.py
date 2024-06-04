import csv
import json
import os.path
import time
import traceback
from datetime import datetime

from rapidfuzz.distance import Levenshtein
from tqdm import tqdm

from nltk.tokenize import word_tokenize

import constant_config
from onto.load import load_mondo_onto_dictionary
from synth_data_gen.prompt_openai import prompt_gpt_35turbo


class LLM_IDS:
    GPT_35 = "GPT_3.5"


def generate_syn_definition_from_concept_name__mondo(onto_dict, llm_id, output_folder):
    """
    Generate a CSV with synthetic definition sentences of concepts from MONDO ontology, by prompting the LLM with ID
    specified as input parameters.
    :param onto_dict: the onto-dictionary loaded by means of the load_mondo_onto_dictionary method present in the onto.load package
    :param llm_id: the ID of the LLM to prompt to generate synthetic data (available LLMs are specified by the class LLM_IDS)
    :param output_folder: the output folder where to store a spreadsheet with synthetic textual data (i.e.e concept definitions)
    """

    # Print stats: how many concepts from MONDO ontology already have a definition?
    mondo_concept_with_def = 0
    mondo_concept_without_def = 0
    # Select concept from MONDO ontology (ID string starting with "MONDO")
    for mondo_concept_id in [cid for cid, c_dict in onto_dict.items() if cid.startswith("MONDO")]:
        if mondo_concept_id in onto_dict and isinstance(onto_dict[mondo_concept_id].definition, str) and len(onto_dict[mondo_concept_id].definition.strip()) > 0:
            mondo_concept_with_def = mondo_concept_with_def + 1
        else:
            mondo_concept_without_def = mondo_concept_without_def + 1
    print(f"MONDO CONCEPTS: {len([cid for cid, c_dict in onto_dict.items() if cid.startswith('MONDO')])} of which "
          f"{mondo_concept_with_def} with a definition and {mondo_concept_without_def} without a definition.")

    # Generate synthetic definition (GPT-3.5-temperature temperature 0.0)
    date_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_csv_file_path__definition = os.path.join(output_folder, f'{date_time}__MONDO__DEFINITION__synthetic_from_concept_synonyms__{llm_id}.csv')
    print(f'Storing definition synthetic data results to CSV file: {output_csv_file_path__definition}...')
    with open(output_csv_file_path__definition, 'w', newline='', encoding='utf-8') as csv_file__definition:

        writer_definition = csv.writer(csv_file__definition, delimiter=',')
        writer_definition.writerow(['CONCEPT ID', 'CONCEPT MAIN NAME', 'LABEL', 'LABEL TYPE', 'SYNTHETIC DEFINITION', 'REAL DEFINITION', 'LLM ID', 'PROMPT', 'ERROR'])
        csv_file__definition.flush()

        for mondo_concept_id in tqdm([cid for cid, c_dict in onto_dict.items() if cid.startswith("MONDO")], desc="Generating synthetic definitions..."):
            # Retrieve concept main name
            mondo_concept_main_name = onto_dict[mondo_concept_id].main_label.strip() if mondo_concept_id in onto_dict and isinstance(onto_dict[mondo_concept_id].main_label, str) and len(onto_dict[mondo_concept_id].main_label.strip()) > 0 else None
            # Retrieve concept exact synonyms
            mondo_concept_synonyms = [syn_name.strip() for syn_name, syn_dict in onto_dict[mondo_concept_id].synonyms.items() if 'scope' in syn_dict and syn_dict['scope'] == 'EXACT']
            # Retrieve concept definition
            mondo_concept_real_definition = onto_dict[mondo_concept_id].definition if mondo_concept_id in onto_dict and isinstance(onto_dict[mondo_concept_id].definition, str) and len(onto_dict[mondo_concept_id].definition.strip()) > 0 else None

            # Select the (sub-)set of concept synonyms that have a Levenshtein distance greater than 10 with respect to
            # already considered synonyms
            mondo_concept_synonyms__filtered = list()
            if len(mondo_concept_synonyms) > 0:
                # Remove synonyms that are too similar to already considered ones (Levenshtein distance <= 10)
                mondo_concept_synonyms__filtered__1 = list()
                for mondo_concept_synonym in mondo_concept_synonyms:
                    lower_levenshtein_distance_with_considered_synonym = 100
                    for already_considered_synonym in [mondo_concept_main_name] + mondo_concept_synonyms__filtered__1:
                        lev_distance = Levenshtein.distance(mondo_concept_synonym, already_considered_synonym)
                        if lev_distance < lower_levenshtein_distance_with_considered_synonym:
                            lower_levenshtein_distance_with_considered_synonym = lev_distance
                    if lower_levenshtein_distance_with_considered_synonym > 10:
                        mondo_concept_synonyms__filtered__1.append(mondo_concept_synonym)

                # Exclude synonyms with same lowercase words but ordered distinctly
                mondo_concept_synonyms__filtered__2 = list()
                for mondo_concept_synonym__filtered__1 in mondo_concept_synonyms__filtered__1:
                    tok_list__mondo_concept_synonym__filtered__1 = [el.lower().strip() for el in word_tokenize(mondo_concept_synonym__filtered__1) if sum([1 if c.isalnum() else 0 for c in el]) > 0]
                    tok_list__mondo_concept_synonym__filtered__1.sort()
                    tok__mondo_concept_synonym__filtered__1 = "".join(tok_list__mondo_concept_synonym__filtered__1) if len(tok_list__mondo_concept_synonym__filtered__1) > 0 else ""

                    exclude_mondo_concept_synonym = False
                    for already_considered_synonym in [mondo_concept_main_name] + mondo_concept_synonyms__filtered__2:
                        tok_list__already_considered_synonym = [el.lower().strip() for el in word_tokenize(already_considered_synonym) if sum([1 if c.isalnum() else 0 for c in el]) > 0]
                        tok_list__already_considered_synonym.sort()
                        tok__already_considered_synonym = "".join(tok_list__already_considered_synonym) if len(tok_list__already_considered_synonym) > 0 else ""
                        if Levenshtein.distance(tok__mondo_concept_synonym__filtered__1, tok__already_considered_synonym) < 1:
                            exclude_mondo_concept_synonym = True
                            break

                    if not exclude_mondo_concept_synonym:
                        mondo_concept_synonyms__filtered__2.append(mondo_concept_synonym__filtered__1)

                mondo_concept_synonyms__filtered = mondo_concept_synonyms__filtered__2

            # Start prompting  to get the definition of concept main name and synonyms
            for mondo_concept_synonym in [mondo_concept_main_name] + mondo_concept_synonyms__filtered:
                if mondo_concept_synonym is not None:
                    error_msg = None

                    # Append concept main name to the synonym in case the synonym is < 10 chars and the synonym is not
                    # included inside the concept main name
                    if len(mondo_concept_synonym) < 10 and mondo_concept_synonym not in mondo_concept_main_name:
                        mondo_concept_synonym = mondo_concept_synonym + ", " + mondo_concept_main_name

                    # Generate synthetic definition
                    prompt = None
                    synthetic_definition = None
                    error_msg = None

                    attempt_num = 0
                    while attempt_num < 3:
                        attempt_num = attempt_num + 1
                        try:
                            # Prompt GPT-3.5
                            if llm_id == LLM_IDS.GPT_35:
                                messages = [
                                    {"role": "system", "content": f"You are an expert in clinical and biomedical "
                                                                  f"sciences."},
                                    {"role": "user", "content": f"Could you provide a single sentence with the "
                                                                f"definition of '{mondo_concept_synonym}'?"},
                                    {"role": "assistant", "content": f""}
                                ]
                                prompt = json.dumps(messages)
                                answer_dict = prompt_gpt_35turbo(msg=messages, temp=0.0)
                                synthetic_definition = f'{answer_dict["list_choices"][0]["content"]}'
                                break

                        except Exception as e:
                            error_msg = f"EXCEPTION WHILE PROMPTING - " \
                                        f"generate_syn_definition_clinical_from_concept_name__mondo " \
                                        f"- model: {llm_id}, concept name: {mondo_concept_synonym} ({mondo_concept_id}), " \
                                        f"exception: {e}"
                            print(error_msg)
                            synthetic_definition = error_msg
                            traceback.print_exc()
                            print("WAITING 10 SECONDS...")
                            time.sleep(10)

                    writer_definition.writerow([f'{mondo_concept_id}', f'{mondo_concept_main_name}', f'{mondo_concept_synonym}',
                                     f'{"SYNONYM" if mondo_concept_synonym != mondo_concept_main_name else "MAIN NAME"}',
                                     f'{synthetic_definition}', f'{mondo_concept_real_definition}',
                                     f'{llm_id}', f'{prompt}', f'{error_msg}'])
                    csv_file__definition.flush()


if __name__ == '__main__':
    """
    Script to generate synthetic definition of concept from MONDO ontology by relying on GPT-3.5-turbo prompting
    """

    # Load MONDO ontology dictionary
    mondo_onto_dict = load_mondo_onto_dictionary()

    # Generate synthetic definitions of MONDO concept synonyms, and store them in new file in folder specified by
    # constant_config.SYNTHETIC_DATA_OUTPUT_FOLDER_PATH
    generate_syn_definition_from_concept_name__mondo(mondo_onto_dict, LLM_IDS.GPT_35, constant_config.SYNTHETIC_DATA_OUTPUT_FOLDER_PATH)
