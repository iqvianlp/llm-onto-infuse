import os
import traceback
from datetime import datetime

import chromadb
from typing import cast
from sentence_transformers import SentenceTransformer

import constant_config

chromadb_client = None


def get_chromadb_client(client_sub_folder=None, reset_data=False, persist=True):
    """
    Get the client of Chroma DB embedding database. Data are persisted in the folder at path specified by
    constant_config.CHROMA_DB_PERSIST_FOLDER_PATH
    :param client_sub_folder: if non-empty string, name of the sub-folder of the folder at path specified by
    constant_config.CHROMA_DB_PERSIST_FOLDER_PATH, where to persist the database
    :param reset_data: (default False) if True, delete all the database data in case disk-persistence is active
    :param persist: (default True) if True, persist client to disk
    :return:
    """
    global chromadb_client
    if chromadb_client is None:
        os.environ["ALLOW_RESET"] = "TRUE"

        if persist is True:
            client_folder_path = constant_config.CHROMA_DB_PERSIST_FOLDER_PATH
            if isinstance(client_sub_folder, str) and len(client_sub_folder.strip()) > 0:
                client_sub_folder_path = os.path.join(client_folder_path, client_sub_folder.strip())
                print(f"CHROMA-DB persist path: specified sub-folder name '{client_sub_folder}'")
                if os.path.exists(client_sub_folder_path) and os.path.isdir(client_sub_folder_path):
                    print(f"CHROMA-DB persist path: the persist path folder already exists, reusing folder '{client_sub_folder_path}'")
                    client_folder_path = client_sub_folder_path
                else:
                    print(f"CHROMA-DB persist path: the persist path folder does not exist, creating folder '{client_sub_folder_path}'")
                    os.mkdir(client_sub_folder_path)
                    client_folder_path = client_sub_folder_path
            else:
                print(f"CHROMA-DB persist path: no sub-folder specified for persist path ({client_sub_folder}).")

            print(f"Set-up Chroma DB client with persist-folder: {client_folder_path}...")
            chromadb_client = chromadb.PersistentClient(path=client_folder_path)

            if reset_data is True:
                print(f"Reset / delete all persisted data of Chroma DB client (stored in persist-folder: {client_folder_path})...")
                chromadb_client.reset()
        else:
            print(f"Set-up non-persisted Chroma DB client...")
            chromadb_client = chromadb.Client()

    return chromadb_client


def get_new_collection(coll_name=None, embedding_function=None):
    """
    Create a new Chroma DB collection with name and embedding function specified. HNSW cosine similarity is exploited.
    :param coll_name: name of the new collection
    :param embedding_function: embedding function useful to convert text to embedding, before storing the embedding in
    the chroma DB collection
    :return: reference to the newly created Chroma DB collection
    """
    global chromadb_client
    if not isinstance(coll_name, str) or len(coll_name.strip()) == 0:
        date_time = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        coll_name = f"UNNAMED_COLLECTION_{date_time}"

    print(f"Creating Chroma DB collection with name {coll_name}...")
    if embedding_function is None:
        return chromadb_client.create_collection(name=coll_name, metadata={"hnsw:space": "cosine"})
    else:
        return chromadb_client.create_collection(name=coll_name, metadata={"hnsw:space": "cosine"},
                                                 embedding_function=embedding_function)


def get_collection_stats(coll_ref=None, print_stats=False):
    """
    Return a dictionary with stats (count of items, similarity function, structure of items, etc.) on the Chroma DB collection.
    :param coll_ref: reference to the collection
    :param print_stats: (default False) if True, collection stats are printed to the standard output
    :return:
    """
    stats_dict = dict()

    try:
        stats_dict['name'] = f'{coll_ref.name}'
        coll_count = coll_ref.count()
        stats_dict['count'] = f'{coll_count}'
        stats_dict['sim_function'] = f'{coll_ref.metadata}'
        if coll_count > 0:
            stats_dict['example_element'] = {k: v[0] if isinstance(v, list) else None for k, v in coll_ref.peek().items() if k != 'embeddings'}
    except Exception:
        traceback.print_exc()

    if print_stats is True:
        for k, v in stats_dict:
            print(f"COLL STATS: {k} --> {v}")

    return stats_dict


class CustomEmbeddingFunction(chromadb.EmbeddingFunction):
    """
    Example of custom embedding function to provide to Chroma DB in order to embed text excerpts before indexing
    the embeddings. The cambridgeltl/SapBERT-from-PubMedBERT-fulltext model is exploited here.
    """
    def __init__(self):
        self.str = SentenceTransformer("avsolatorio/GIST-Embedding-v0", trust_remote_code=True)

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        sentence_transformer_ef = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        return cast(
            chromadb.Embeddings,
            self.str.encode(
                list(input),
                convert_to_numpy=True,
                normalize_embeddings=False
            ).tolist(),
        )


if __name__ == '__main__':

    cdb_client = get_chromadb_client()
    cdb_test_coll_ref = get_new_collection(coll_name=None, embedding_function=CustomEmbeddingFunction())

    print(cdb_client.list_collections())
    print(get_collection_stats(coll_ref=cdb_test_coll_ref))

    cdb_test_coll_ref.add(
        documents=["This is a short sentence.", "The sky is blue.", "The sky is gray."],
        metadatas=[{"sent": "1"}, {"sent": "2"}, {"sent": "3"}],
        ids=["id1", "id2", "id3"]
    )
    print(get_collection_stats(coll_ref=cdb_test_coll_ref))

    query_result = cdb_test_coll_ref.query(
        query_texts=["The sky is blue.", "The sky is dark."],
        n_results=10,
        include=["documents", "distances"]
    )
