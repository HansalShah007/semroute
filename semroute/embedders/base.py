import numpy as np
from typing import List, Dict
from semroute.utils.similarity import cosine_similarity

class BaseEmbedder:
    dynamic_threshold = 1.0

    def __init__(self, model_name: str, valid_model_names: List[str], static_thresholds: Dict[str, float]):
        """
        Initializes the embedder with the specified model name.

        Parameters:
        - model_name (str): The name of the embedding model to use.
        - valid_model_names (List[str]): A list of valid model names.
        - static_thresholds (Dict[str, float]): A dictionary of static thresholds for each model.

        Raises:
        - ValueError: If the model name is not one of the supported models.
        """
        if model_name not in valid_model_names:
            raise ValueError(f"Incorrect embedding model name, choose from {valid_model_names}")

        self.model_name = model_name
        self.static_thresholds = static_thresholds

    def embed_utterances(self, utterances: List[str]) -> np.ndarray:
        """
        Embeds a list of utterances into their respective embeddings using the specified model.

        Parameters:
        - utterances (List[str]): A list of utterances to embed.

        Returns:
        - np.ndarray: A numpy array of embeddings corresponding to the input utterances.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    def adapt_threshold(self, similar_utterance_embeds: np.ndarray, original_utterance_embeds: np.ndarray):
        """
        Adapts the dynamic threshold based on the embeddings of similar utterances and the original utterances.

        Parameters:
        - similar_utterance_embeds (np.ndarray): A numpy array of embeddings for similar utterances.
        - original_utterance_embeds (np.ndarray): A numpy array of embeddings for the original utterances.
        """
        for sim_utter_embed in similar_utterance_embeds:
            avg_threshold = 0
            for org_utter_embed in original_utterance_embeds:
                avg_threshold += cosine_similarity(sim_utter_embed, org_utter_embed)
            avg_threshold /= len(original_utterance_embeds)
            self.dynamic_threshold = min(self.dynamic_threshold, avg_threshold)

    def get_static_threshold_score(self) -> float:
        """
        Returns the static threshold score for the configured embedding model.

        Returns:
        - float: The static threshold score.
        """
        return self.static_thresholds[self.model_name]
