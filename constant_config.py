import os.path

# Full local path of the directory where all files required / created by the project are saved
BASE_DATA_FOLDER_PATH = "/Users/u1108046/Downloads/TEST_PACKAGE"

# Name of the MONDO ontology pickle object file (exploited to store the contents of the ontology to support
# ontological knowledge infusion)
MONDO_ONTO_PICKLE_FILE = "mondo_concept_dict.pk"
# Full local path of the MONDO ontology pickle object file
MONDO_ONTO_PICKLE_FILE_PATH = os.path.join(BASE_DATA_FOLDER_PATH, MONDO_ONTO_PICKLE_FILE)

# Name of the evaluation datasets pickle object file (exploited to store the contents of the - sentence similarity -
# evaluation datasets, useful to quantify the effectiveness of ontological knowledge infusion)
EVAL_DATA_PICKLE_FILE = "eval_data.pk"
# Full local path of the evaluation dataset pickle object file
EVAL_DATA_PICKLE_FILE_PATH = os.path.join(BASE_DATA_FOLDER_PATH, EVAL_DATA_PICKLE_FILE)

# Full local path of the folder where synthetic data for ontological knowledge infusion are stored to / loaded from
SYNTHETIC_DATA_OUTPUT_FOLDER_PATH = os.path.join(BASE_DATA_FOLDER_PATH, "ONTO_FUSE_SYNTH_DATA")

# Full local path of the persist-directory of the Chroma DB embedding database
CHROMA_DB_PERSIST_FOLDER_PATH = os.path.join(BASE_DATA_FOLDER_PATH, "CHROMA_DB")

# Full local path of the directory where evaluation results of ontological-knowledge-infused embedding-LLMs are stored
EVALUATION_RESULT_PATH = os.path.join(BASE_DATA_FOLDER_PATH, "EVAL_RESULT")

# Name of the OpenAI completion cache pickle object file useful to store OpenAI LLM prompt-answer pairs)
OPENAI_CACHE_PICKLE_FILE = "completion_cache_file__OpenAI_ontoInfuse.pk"
# Full local path of the OpenAI completion cache pickle object file
OPENAI_CACHE_PICKLE_FILE_PATH = os.path.join(BASE_DATA_FOLDER_PATH, OPENAI_CACHE_PICKLE_FILE)