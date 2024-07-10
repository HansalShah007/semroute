import numpy as np
from typing import List, Callable
from semroute.utils.similarity import cosine_similarity

class CustomEmbedder:
    dynamic_threshold = 1.0

    def __init__(
            self, 
            custom_embedding_function: Callable[[List[str]], np.ndarray], 
            static_threshold: float, 
            vector_embedding_size: int):
        """
        Initializes the embedder with the specified embedding function and static threshold.

        Parameters:
        - custom_embedding_function (Callable[[List[str]], np.ndarray]): The custom embedding function.
        - static_threshold (float): The static threshold for the embeddings.

        Raises:
        - ValueError: If the custom embedding function does not have the correct signature.
        """
        
        # Testing the custom_embedding_function with a sample input
        test_input = ["test"]
        try:
            test_output = custom_embedding_function(test_input)
        except Exception as e:
            raise ValueError("The custom embedding function's signature is incorrect. It should take only one argument of type List[str] as input")
        
        if not isinstance(test_output, np.ndarray):
            raise ValueError("The custom embedding function must return a numpy array")
        if test_output.shape != (len(test_input), vector_embedding_size):
            raise ValueError(f"The custom embedding function should return a numpy array of size (number_of_utterances, {vector_embedding_size}), it returned an array of size: {test_output.shape}")
        
        self.custom_embedding_function = custom_embedding_function
        self.static_threshold = static_threshold

    def embed_utterances(self, utterances: List[str]) -> np.ndarray:
        """
        Embeds a list of utterances into their respective embeddings using the specified model.

        Parameters:
        - utterances (List[str]): A list of utterances to embed.

        Returns:
        - np.ndarray: A numpy array of embeddings corresponding to the input utterances.
        """
        return self.custom_embedding_function(utterances)

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
        return self.static_threshold