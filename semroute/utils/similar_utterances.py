import json
import os
from openai import OpenAI
from typing import List

class OpenAIChatAPIError(Exception):
    pass

def openai_chat_api(
    messages, model='gpt-3.5-turbo', max_tokens=4095, temperature=1, seed=42
) -> str:
    """
    Calls the OpenAI Chat API with the provided messages and parameters.

    Parameters:
    - messages: The list of message dictionaries to send to the OpenAI API.
    - model (str): The model to use for the API call (default is 'gpt-3.5-turbo').
    - temperature (int): The temperature setting for the model's responses (default is 1).
    - seed (int): The seed for the model's responses to ensure consistency (default is 42).

    Returns:
    - str: The content of the response message from the OpenAI API.

    Raises:
    - ValueError: If the OpenAI API key is not set in the environment.
    """

    try: 
        client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        response = client.chat.completions.create(
            response_format={ "type": "json_object" },
            messages=messages,
            model=model,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        if e == 'The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable':
            raise ValueError("Dynamic thresholding type uses an OpenAI LLM and it needs an OPENAI_API_KEY defined in the environment.")

def get_similar_utterances(
    utterances: List[str],
    description: str,
    retries: int = 3
) -> List[str]:
    """
    Generates a list of similar utterances based on the input utterances and description
    using the OpenAI Chat API. This is used for dynamic thresholding to adapt the threshold
    based on similar examples.

    Parameters:
    - utterances (List[str]): A list of example utterances.
    - description (str): A description to generate similar utterances.

    Returns:
    - List[str]: A list of similar utterances.
    """

    system_message = f"""
I will give you a list of utterances. These utterances are going to be used for semantic routing. I want you to generate similar utterances that can be used to corroborate the given utterances for a better performance in the semantic routing. You will also be given a description for the set of utterances so that you can better understand the intent of the user for semantic routing.

Output a JSON string as follows:
{{
    "similar_utterances" = [<List of string of utterances>]
}}

Generate a list of {min(max(len(utterances)*2, 50), 200)} similar utterances.
"""

    user_message = f"""
Original Utterances:
{utterances}

Description:
{description}
"""
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    
    llm_resp = openai_chat_api(messages)
    try:
        response_json = json.loads(llm_resp)
        if "similar_utterances" not in response_json:
            raise ValueError("The response does not contain 'similar_utterances'")
        similar_utterances = response_json["similar_utterances"]
    except (json.JSONDecodeError, ValueError) as e:
        if retries > 0:
            return get_similar_utterances(utterances, description, retries-1)
        else:
            raise OpenAIChatAPIError("Failed to generate similar utterances after multiple retries") from e

    return similar_utterances