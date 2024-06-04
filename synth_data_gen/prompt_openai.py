import atexit
import hashlib
import json
import os
import pickle
import traceback
from datetime import datetime
from pathlib import Path

import openai
from azure.identity import ClientSecretCredential

import constants as const

from constant_config import OPENAI_CACHE_PICKLE_FILE_PATH


class OpenAIprompting():
    completion_cache_path = OPENAI_CACHE_PICKLE_FILE_PATH
    completion_cache_dict = None

    openai_token_timestamp = None

    @staticmethod
    def request_openai_token(force=False):
        """
        Set-up credentials to access OpenAI / GPT
        :param force: set tot True to force credentials / token update
        :return:
        """
        is_valid = True
        if OpenAIprompting.openai_token_timestamp is None:
            print('OpenAI token not yet requested')
            is_valid = False
        elif (datetime.now() - OpenAIprompting.openai_token_timestamp).total_seconds() > 60 * 45:
            print('OpenAI token expired')
            is_valid = False

        if force is True or is_valid is False:
            # Request OpenAI token with team's credentials (in constants.py)
            print("Requesting OpenAI token...")

            # Dumping cache
            print("(storing a dump of completion cache)")
            OpenAIprompting.dump_completion_cache()

            # Authenticate to Azure
            credentials = ClientSecretCredential(const.TENANT_ID, const.SERVICE_PRINCIPAL,
                                                 const.SERVICE_PRINCIPAL_SECRET)
            token = credentials.get_token(const.SCOPE_NON_INTERACTIVE)
            # Access openai account
            openai.api_type = "AO_ADD"
            openai.api_key = token.token
            openai.api_base = f"{const.OPENAI_API_BASE}/{const.OPENAI_API_TYPE}/{const.OPENAI_ACCOUNT_NAME}"
            openai.api_version = const.OPENAI_API_VERSION

            OpenAIprompting.openai_token_timestamp = datetime.now()

    @staticmethod
    def dump_completion_cache():
        """
        Dump to file the completion_cache_dict, caching LLM calls
        """
        if OpenAIprompting.completion_cache_dict is not None:
            Path(OpenAIprompting.completion_cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(OpenAIprompting.completion_cache_path, 'wb') as f:
                pickle.dump(OpenAIprompting.completion_cache_dict, f)

    @staticmethod
    def set_up_completion_cache():
        """
        Set up the completion_cache_dict, caching LLM calls
        :return:
        """
        if OpenAIprompting.completion_cache_dict is None:
            cache_dump_file = Path(OpenAIprompting.completion_cache_path)
            if cache_dump_file.is_file():
                with open(OpenAIprompting.completion_cache_path, 'rb') as f:
                    OpenAIprompting.completion_cache_dict = pickle.load(f)
                    print(f"Loaded GPT call cache dict from file {OpenAIprompting.completion_cache_path} - "
                          f"{len(OpenAIprompting.completion_cache_dict)} cached calls.")
            else:
                OpenAIprompting.completion_cache_dict = dict()
                print(f"IMPOSSIBLE TO LOAD GPT call cache dict from file {OpenAIprompting.completion_cache_path} - "
                      f"{len(OpenAIprompting.completion_cache_dict)} cached calls.")

            atexit.register(OpenAIprompting.dump_completion_cache)


def prompt_gpt_35turbo(msg, temp=0.0):
    """
    Prompt GPT-3.5-turbo to generate text by relying on the prompt dialogue messages and temperature specified as
    parameters
    :param msg: list of prompt dialogue messages - [ {"role": "system", "content": "....."}, {"role": "user", "content": "....."}, ... ]
    :param temp: temperature for text generation
    :return: dictionary with prompt info and generated text
    """
    OpenAIprompting.set_up_completion_cache()

    # deployment ID (LLM) to rely on to generate text
    deployment_id = "gpt-35-turbo-0613"

    generated_text_dict = dict()

    # Check if the generated text is available from the OpenAI cache
    prompt_interaction_hash = f"{deployment_id}__{temp}__" + hashlib.md5(json.dumps(msg).encode('utf-8')).hexdigest()

    if prompt_interaction_hash in OpenAIprompting.completion_cache_dict:
        print("Result retrieved from cache.")
        generated_text_dict = OpenAIprompting.completion_cache_dict[prompt_interaction_hash]
    else:
        # Authenticate and, if needed, renew token
        OpenAIprompting.request_openai_token()

        completion = openai.ChatCompletion.create(
            deployment_id=deployment_id,
            messages=msg,
            temperature=temp
        )

        # Populate generated_text_dict
        try:
            generated_text_dict = {
                'input_msg': msg,
                'input_temperature': temp,
                'input_deployment': deployment_id,
                'original_object': completion,
                'engine': completion.engine,
                'usage_total': completion.usage.total_tokens,
                'usage_prompt': completion.usage.prompt_tokens,
                'usage_completion': completion.usage.completion_tokens,
                'num_choices': len(completion.choices),
                'list_choices': list()
            }
            for choice_idx, choice in enumerate(completion.choices):
                if not hasattr(choice.message, 'content') or not isinstance(choice.message.content, str):
                    print(">>>>>>> ATTENTION!!! <<<<<<<")
                    print("Impossible to retrieve message content of message:")
                    print(f"{choice}")
                    print(f"--- COMPLETION:")
                    print(f"{completion}")
                    print(">>>>>>> ATTENTION!!! <<<<<<<")

                generated_text_dict['list_choices'].append({
                    'role': choice.message.role,
                    'content': choice.message.content if hasattr(choice.message, 'content') and isinstance(choice.message.content, str) else ''
                })
        except:
            print(f"EXCEPTION WHILE EXTRACTING INFO FROM LLM RESPONSE - {msg}")
            traceback.print_exc()

        # Store in OpenAI cache
        if len(generated_text_dict) > 0 and 'engine' in generated_text_dict:
            OpenAIprompting.completion_cache_dict[prompt_interaction_hash] = {k: v for k, v in
                                                                              generated_text_dict.items()
                                                                              if k != 'original_object'}

    return generated_text_dict


if __name__ == '__main__':
    messages = [
        {"role": "system", "content": "You are an expert in biomedicine."},
        {"role": "user", "content": f"Could you provide the definition of type II diabetes mellitus in one sentence?"},
        {"role": "assistant", "content": f""}
    ]
    gen_text_dict = prompt_gpt_35turbo(msg=messages)
    print(f"Generated text: {gen_text_dict}")
