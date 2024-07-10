import os
import numpy as np
from semroute.embedders.base import BaseEmbedder
from typing import List
from openai import OpenAI

class OpenAIEmbedder(BaseEmbedder):
    """
    A class to interact with OpenAI's embedding models for embedding utterances.
    
    Inherits from:
    - BaseEmbedder: The base class for embedding models.
    """
    
    def __init__(self, model_name: str = "text-embedding-3-small"):

        """
        Initializes the OpenAIEmbedder with the specified model name and sets up the OpenAI client.
        
        Parameters:
        - model_name (str): The name of the OpenAI embedding model to use (default is "text-embedding-3-small").

        Raises:
        - ValueError: If the OpenAI API key is not provided in the environment.
        - ValueError: If the model name is not one of the supported models.
        """
        
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OpenAI API key not provided")
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

        static_thresholds = {
            'text-embedding-3-small': 0.22,
            'text-embedding-3-large': 0.22,
            'text-embedding-ada-002': 0.75
        }
        valid_model_names = ['text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002']
        super().__init__(model_name, valid_model_names, static_thresholds)


    def embed_utterances(self, utterances: List[str]) -> np.ndarray:
        """
        Embeds a list of utterances into their respective embeddings using the specified OpenAI model.

        Parameters:
        - utterances (List[str]): A list of utterances to embed.

        Returns:
        - np.ndarray: A numpy array of embeddings corresponding to the input utterances.
        """
        utterances = [uttr.lower() for uttr in utterances]

        embedddings = []

        total_words = 0
        utter_to_embed = []
        for utter in utterances:
            if total_words >= 6000:
                embedddings.extend([data.embedding for data in self.client.embeddings.create(input=utter_to_embed, model=self.model_name).data])
                total_words = 0
                utter_to_embed = []
            else:
                utter_to_embed.append(utter)
                total_words += len(utter.split())
        
        if len(utter_to_embed):
            embedddings.extend([data.embedding for data in self.client.embeddings.create(input=utter_to_embed, model=self.model_name).data])

        embeddings = np.array(embedddings)
        return embeddings
