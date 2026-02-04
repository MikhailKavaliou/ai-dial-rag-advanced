
import requests

DIAL_EMBEDDINGS = 'https://ai-proxy.lab.epam.com/openai/deployments/{model}/embeddings'


class DialEmbeddingsClient:
    """Client for DIAL embeddings API"""

    def __init__(self, deployment_name: str, api_key: str):
        """
        Initialize the embeddings client

        Args:
            deployment_name: The name of the deployment/model to use
            api_key: The API key for authentication
        """
        if not api_key or api_key.strip() == "":
            raise ValueError("API key cannot be null or empty")

        self._endpoint = DIAL_EMBEDDINGS.format(model=deployment_name)
        self._api_key = api_key

    def get_embeddings(self, inputs: list[str], dimensions: int = 1536) -> dict[int, list[float]]:
        """
        Generate embeddings for a list of input texts

        Args:
            inputs: List of text strings to generate embeddings for
            dimensions: The dimension of the embeddings (default: 1536)

        Returns:
            Dictionary mapping input index to embedding vector
        """
        headers = {
            "api-key": self._api_key,
            "Content-Type": "application/json"
        }

        request_data = {
            "input": inputs,
            "dimensions": dimensions
        }

        response = requests.post(
            url=self._endpoint,
            headers=headers,
            json=request_data,
            timeout=60
        )

        if response.status_code == 200:
            data = response.json()
            embeddings_data = data.get("data", [])

            # Create a dictionary mapping index to embedding vector
            result = {}
            for item in embeddings_data:
                index = item.get("index")
                embedding = item.get("embedding")
                result[index] = embedding

            return result
        else:
            raise Exception(f"HTTP {response.status_code}: {response.text}")


# Hint:
#  Response JSON:
#  {
#     "data": [
#         {
#             "embedding": [
#                 0.19686688482761383,
#                 ...
#             ],
#             "index": 0,
#             "object": "embedding"
#         }
#     ],
#     ...
#  }
