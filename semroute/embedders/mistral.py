import os
import numpy as np
from semroute.embedders.base import BaseEmbedder
from typing import List
from mistralai.client import MistralClient

class MistralAIEmbedder(BaseEmbedder):
    """
    A class to interact with Mistral AI's embedding models for embedding utterances.
    
    Inherits from:
    - BaseEmbedder: The base class for embedding models.
    """

    def __init__(self, model_name: str = "mistral-embed"):
        """
        Initializes the MistralAIEmbedder with the specified model name and sets up the Mistral AI client.
        
        Parameters:
        - model_name (str): The name of the Mistral AI embedding model to use (default is "mistral-embed").

        Raises:
        - ValueError: If the Mistral API key is not provided in the environment.
        """

        api_key = os.getenv("MISTRALAI_API_KEY")
        if api_key is None:
            raise ValueError("Mistral API key not provided")
        self.client = MistralClient(api_key=os.environ['MISTRALAI_API_KEY'])
        
        static_thresholds = {'mistral-embed': 0.6}
        valid_model_names = ['mistral-embed']
        super().__init__(model_name, valid_model_names, static_thresholds)

    def embed_utterances(self, utterances: List[str]) -> np.ndarray:
        """
        Embeds a list of utterances into their respective embeddings using the specified Mistral AI model.

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
                embedddings.extend([data.embedding for data in self.client.embeddings(input=utter_to_embed, model=self.model_name).data])
                total_words = 0
                utter_to_embed = []
            else:
                utter_to_embed.append(utter)
                total_words += len(utter.split())
        
        if len(utter_to_embed):
            embedddings.extend([data.embedding for data in self.client.embeddings(input=utter_to_embed, model=self.model_name).data])

        embeddings = np.array(embedddings)
        return embeddings