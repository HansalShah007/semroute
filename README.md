# SemRoute

SemRoute is a semantic router that enables routing using the semantic meaning of queries. This tool leverages vector embeddings to make decisions quickly, without the need to train a classifier or call a Large Language Model. SemRoute is simple to use and offers flexibility in choosing different embedding models, thresholding types, and scoring methods to best fit your use case.

## Installation

Install the library using the command:

```bash
pip install semroute
```
[PyPI Package](https://pypi.org/project/semroute/)

To use the semantic router, you need to create a **Router** and add semantic routes that will define the available routes for a given query.

## Usage
### Creating a `Router`

```python
from semroute import Route

router = Router(
	embedder_host="OpenAI",
	embedder_model="text-embedding-3-large",
	thresholding_type="dynamic",
	scoring_method="centroid"
)
```

**Configuration Options**

1. `embedder_host`: SemRoute currently supports embedding models from `OpenAI` and `MistralAI`. So, you can choose either of them for using their embedding models.

2. `embedder_model`: This field can be used for specifying the embedding model to be used from the `embedder_host`. Given below are the embedding models supported from each host:
	- **OpenAI**: \[`text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002`\]
	- **MistralAI**: \[`mistral-embed`\]

3. `thresholding_type`: This field specifies the type of thresholding mechanism to be used for routing the queries. SemRoute supports two types of thresholding:
	- `static`: This type instructs the router to use the preset thresholds for each embedding model to determine if the query belongs to a specific route or not. It is model dependent and these thresholds are refined for each embedding model. It leads to faster routing decision because there is no other overhead other than generating the embeddings of the provided utterances. However, using this type of thresholding can sometimes lead to wrong routing decisions because it is not adapted for the sample utterances that you provide for the route.
	- `dynamic`: This thresholding type instructs the Router to adapt the threshold for the embedding model using the sample utterances provided in the route. For using this mode, you need to provide `OPENAI_API_KEY` and set it as your environment variable. This mode uses OpenAI's `GPT-3.5-Turbo` to generate more utterances similar to that provided by the user and uses them to fine-tune the dynamic threshold. This method is slower but leads to more accurate routing decisions.

4. `scoring_method`: This field is used for specifying the method used for scoring how similar the query is to each route. SemRoute supports two scoring methods:
	- `individual_averaging`: In this method, similarity score is calculated between each utterance embedding of the route and the query. Then the average of these similarities is used for making a routing decision. This method has a time complexity of `O(n)`.
	- `centroid`: In this method, a centroid is calculated for each route using the individual utterance embeddings and then the similarity between this centroid and the query embedding is used for making a routing decision. This method has a time complexity of `O(1)`.

### Custom Embedding Models

SemRoute now supports using custom embedding models. You can provide your own embedding function along with its static threshold and vector embedding size. If a custom embedding function is provided, it will take precedence over the preconfigured ones.

Here is the schema that the user should follow for the custom embedding configuration:

- **`embedding_function`**: A callable function that takes a list of strings and returns a numpy array of embeddings.
- **`static_threshold`**: A float value representing the static threshold for routing decisions.
- **`vector_embedding_size`**: An integer representing the size of the vector embeddings.

**Example configuration**

```python
custom_embedder_config = {
    "embedding_function": your_custom_embedding_function,
    "static_threshold": 0.5,
    "vector_embedding_size": 768
}

router = Router(
    custom_embedder=custom_embedder_config,
    thresholding_type="static",
    scoring_method="centroid"
)
```

**Example custom embedding function**

Your custom embedding function should adhere to the following format:

```python
def your_custom_embedding_function(utterances: List[str]) -> np.ndarray:
    # Your logic to convert utterances to embeddings
    embeddings = np.array([your_embedding_logic(utterance) for utterance in utterances])
    return embeddings

```

In this example, replace `your_embedding_logic` with the logic specific to your embedding model.

By providing a custom embedding configuration, you can integrate any embedding model into SemRoute, making it highly flexible and adaptable to various use cases.
### Adding `routes`

```python
router.add_route(
    name="technology",
    utterances=[
        "what's the latest in tech news?",
        "tell me about artificial intelligence",
        "how does blockchain work?",
        "what is the best programming language?",
        "can you recommend a good laptop?",
        "what's new with the iPhone?"
    ],
    description="A group of utterances for when the user discusses anything related to technology"
)

router.add_route(
    name="sports",
    utterances=[
        "who won the game last night?",
        "what's the score of the basketball game?",
        "tell me about the latest in football",
        "who's your favorite athlete?",
        "do you think they'll win the championship?",
        "when is the next World Cup?"
    ],
    description="A group of utterances for when the user discusses anything related to sports"
)

router.add_route(
    name="food",
    utterances=[
        "what's your favorite food?",
        "can you recommend a good restaurant?",
        "how do you make spaghetti?",
        "what's a good recipe for a healthy dinner?",
        "tell me about the best dessert you've had",
        "what's your favorite cuisine?"
    ],
    description="A group of utterances for when the user discusses anything related to food"
)

router.add_route(
    name="travel",
    utterances=[
        "where's the best place to travel?",
        "can you recommend a vacation spot?",
        "what's the best way to travel on a budget?",
        "tell me about your favorite trip",
        "where should I go for my next holiday?",
        "what are the top tourist destinations?"
    ],
    description="A group of utterances for when the user discusses anything related to travel"
)

router.add_route(
    name="health",
    utterances=[
        "what's the best way to stay healthy?",
        "can you recommend a good workout?",
        "tell me about a healthy diet",
        "how do I reduce stress?",
        "what are the benefits of meditation?",
        "how do I improve my mental health?"
    ],
    description="A group of utterances for when the user discusses anything related to health"
)
```


> For better routing decisions make sure to include as many cases of utterance as possible for each route. This will help the router to ensure that no edge case is left out while making a routing decision. Also, while using the `dynamic` mode, please make sure to give a `description` that very closely aligns with the intent of that route because it is used for generating similar utterances.

### To make a routing decision

```python
router.route("How much does the health insurance costs?")
```

```shell
[OUT]: health
```

```python
router.route("Let's go to Italy!")
```

```shell
[OUT]: travel
```

### To save/ load your router configuration

The tools allows you to save your configured router into a pickle file using the `save_router` method.

```python
router.save_router("path/to/filename.pkl")
```

You can also load your saved configuration and use it to configure a router.

```python
from semroute import Router

router = Router()
router.load_router("path/to/filename.pkl")
router.route("Query to route")
```

## Contribution

Contributions to SemRoute are welcome! Please ensure your pull requests are well-documented and tested.
## License

SemRoute is licensed under the MIT License. See the LICENSE file for more details.

For any issues or feature requests, please open an issue on the [GitHub repository](https://github.com/HansalShah007/semroute).