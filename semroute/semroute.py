import pickle
import numpy as np
import sys
import logging
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, ValidationError, model_validator
from semroute.embeders.openai import OpenAIEmbeder
from semroute.embeders.mistral import MistralAIEmbeder
from semroute.utils.centroid import get_centroid
from semroute.utils.similar_utterances import get_similar_utterances
from semroute.utils.similarity import cosine_similarity

logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

embedding_models = {
    "OpenAI": ['text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002'],
    "MistralAI": ['mistral-embed']
}

host_embeder_class_mapping = {
    "OpenAI": OpenAIEmbeder,
    "MistralAI": MistralAIEmbeder
}


class RouteSchema(BaseModel):
    name: str
    utterances: List[str]
    utterance_embeddings: np.ndarray
    description: str
    centroid: Optional[np.ndarray] = None

    class Config:
        arbitrary_types_allowed = True


class RouterSchema(BaseModel):
    embeder_host: Literal['OpenAI', 'MistralAI']
    embeder_model_name: str
    thresholding_type: Literal['static', 'dynamic']
    scoring_method: Literal['individual_averaging', 'centroid']
    dynamic_threshold: Optional[float] = None
    routes: List[RouteSchema]

    @model_validator(mode='after')
    def check_embedding_model(self):
        embeder_host = self.embeder_host
        embeder_model_name = self.embeder_model_name
        if embeder_model_name not in embedding_models[embeder_host]:
            raise ValueError(f"Embeder model is not supported by the embeder host, choose from {embedding_models[embeder_host]}")
        return self

    @model_validator(mode='after')
    def check_dynamic_threshold(self):
        thresholding_type = self.thresholding_type
        dynamic_threshold = self.dynamic_threshold
        if thresholding_type == 'dynamic' and dynamic_threshold is None:
            raise ValueError(
                f"For dynamic thresholding, there must be a dynamic threshold to be used in the configuration")
        return self

    @model_validator(mode='after')
    def check_centroids(self):
        scoring_method = self.scoring_method
        routes = self.routes
        if scoring_method == 'centroid':
            for route in routes:
                if route.centroid is None:
                    raise ValueError(
                        f"For centroid scoring method, there must be a centroid field for all the routes")
        return self


class Router:

    routes: List[Dict] = []

    def __init__(
        self,
        embeder_host: str = "OpenAI",
        embeder_model: str = "text-embedding-3-small",
        thresholding_type: str = "static",
        scoring_method: str = "individual_averaging"
    ):
        """
        Initializes the Router with the specified embedding host, model, thresholding type, and scoring method.

        Parameters:
        - embeder_host (str): The host providing the embedding model (e.g., "OpenAI" or "MistralAI").
        - embeder_model (str): The specific embedding model to use (e.g., "text-embedding-3-small").
        - thresholding_type (str): The type of thresholding to use ("static" or "dynamic").
        - scoring_method (str): The method used for scoring the similarity ("individual_averaging" or "centroid").

        Raises:
        - ValueError: If the `embeder_host`, `embeder_model`, `thresholding_type`, or `scoring_method` is invalid.
        """

        if embeder_host not in ['OpenAI', 'MistralAI']:
            raise ValueError(
                "Incorrect value for embeder host, choose from ['OpenAI', 'MistralAI']")
        self.embeder_host = embeder_host

        if embeder_model not in embedding_models[embeder_host]:
            raise ValueError(f"Incorrect value for embeder model, choose from {embedding_models[embeder_host]}")
        self.embeder_model_name = embeder_model
        self.embeder_model = host_embeder_class_mapping[self.embeder_host](
            embeder_model)

        if thresholding_type not in ["static", "dynamic"]:
            raise ValueError(
                "Incorrect value for thresholding type, choose from ['static', 'dynamic']")
        self.thresholding_type = thresholding_type

        if scoring_method not in ['individual_averaging', 'centroid']:
            raise ValueError(
                "Incorrect value for scoring method, choose from ['individual_averaging', 'centroid']")
        self.scoring_method = scoring_method

    def add_route(
        self,
        name: str,
        utterances: List[str],
        description: str
    ):
        """
        Adds a new route to the router with the specified name, list of example utterances, and a description.
        Embeds the utterances and optionally computes the centroid for the route based on the scoring method.

        Parameters:
        - name (str): The name of the route.
        - utterances (List[str]): A list of example utterances for the route.
        - description (str): A description of the route.

        Raises:
        - ValueError: If the list of utterances is empty.
        """

        if len(utterances):

            route = {
                "name": name,
                "utterances": utterances,
                "utterance_embeddings": self.embeder_model.embed_utterances(utterances),
                "description": description
            }

            if self.scoring_method == 'centroid':
                route['centroid'] = get_centroid(route['utterance_embeddings'])

            if self.thresholding_type == 'dynamic':
                similar_utterances = get_similar_utterances(
                    utterances, description)
                similar_embeddings = self.embeder_model.embed_utterances(
                    similar_utterances)
                self.embeder_model.adapt_threshold(
                    similar_embeddings, route['utterance_embeddings'])

            self.routes.append(route)

        else:

            raise ValueError("The list of utterances cannot be empty")

    def route(
        self,
        query: str
    ) -> str:
        """
        Routes the input query into one of the known routes based on the configured embedding model and scoring method.
        Returns the name of the best matching route or `None` if no route meets the threshold.

        Parameters:
        - query (str): The input query string to route.

        Returns:
        - str: The name of the best matching route or `None` if no match is found.
        """

        if len(self.routes):
            query_embedding = self.embeder_model.embed_utterances([query])[0]

            if self.thresholding_type == 'static':
                threshold = self.embeder_model.get_static_threshold_score()
            elif self.thresholding_type == 'dynamic':
                threshold = self.embeder_model.dynamic_threshold
            route_scores = {}
            if self.scoring_method == 'individual_averaging':

                for route in self.routes:
                    avg_score = 0
                    for i, utter_embed in enumerate(route['utterance_embeddings']):
                        avg_score += cosine_similarity(query_embedding,
                                                    utter_embed)
                    avg_score /= len(route['utterance_embeddings'])
                    if avg_score >= threshold:
                        route_scores[route['name']] = avg_score

            elif self.scoring_method == 'centroid':

                for route in self.routes:
                    score = cosine_similarity(query_embedding, route['centroid'])
                    if score >= threshold:
                        route_scores[route['name']] = score

            if route_scores:
                sort_routes = list(route_scores.items())
                sort_routes.sort(key=lambda x: x[1], reverse=True)
                return sort_routes[0][0]
            else:
                return None

        else:
            logging.error("Routes are missing! Add routes to the router for routing a query")
            sys.exit(1)
            
    def save_router(
        self,
        filepath: str
    ):
        """
        This function saves the routes specific to the Router instance in a pickle file.
        """

        if filepath[-3:] != 'pkl':
            raise ValueError(
                "The filepath should lead to a path where router configuration needs to be stored and the filename should be ending with a 'pkl' extension")

        router = {
            "embeder_host": self.embeder_host,
            "embeder_model_name": self.embeder_model_name,
            "thresholding_type": self.thresholding_type,
            "scoring_method": self.scoring_method,
            "routes": self.routes
        }

        if self.thresholding_type == 'dynamic':
            router['dynamic_threshold'] = self.embeder_model.dynamic_threshold

        with open(f'{filepath}', 'wb') as f:
            pickle.dump(router, f)

    def load_router(
        self,
        filepath: str
    ):
        """
        This function loads the pre-configured routes from a pickle file.
        """

        if filepath[-3:] != 'pkl':
            raise ValueError(
                "The filepath should lead to a pickle file ending with a 'pkl' extension")

        with open(filepath, 'rb') as f:
            preconfig_routes = pickle.load(f)

        try:
            RouterSchema(**preconfig_routes)

            # Configuring the router
            self.embeder_host = preconfig_routes['embeder_host']
            self.embeder_model_name = preconfig_routes['embeder_model_name']
            self.embeder_model = host_embeder_class_mapping[self.embeder_host](
                self.embeder_model_name)
            self.thresholding_type = preconfig_routes['thresholding_type']
            self.scoring_method = preconfig_routes['scoring_method']

            if self.thresholding_type == 'dynamic':
                self.embeder_model.dynamic_threshold = preconfig_routes['dynamic_threshold']

            self.routes = preconfig_routes['routes']

        except ValueError or ValidationError as e:
            logging.error(e)
            sys.exit(1)
