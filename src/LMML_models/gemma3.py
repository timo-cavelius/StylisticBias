"""
Gemma 3 judge using Google Cloud Vertex AI deployed model endpoint.
"""

import base64
import os
from pathlib import Path

from google.cloud import aiplatform

from .base import JudgeModel


class Gemma3Judge(JudgeModel):
    def __init__(
        self,
        endpoint_id: str,
        project_id: str | None = None,
        location: str = "us-central1",
        temperature: float = 0.2,
    ) -> None:
        """Initialize Gemma 3 judge using Vertex AI deployed endpoint.
        
        Args:
            endpoint_id: Vertex AI endpoint ID or full resource name
            project_id: Google Cloud project ID. If None, uses GOOGLE_CLOUD_PROJECT_ID env var
            location: Google Cloud location (default: us-central1)
            temperature: Temperature for generation (default: 0.2)
        """
        self.endpoint_id = endpoint_id
        self.temperature = temperature
        
        if project_id is None:
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
            if not project_id:
                raise ValueError("project_id must be provided or GOOGLE_CLOUD_PROJECT_ID env var must be set")
        
        self.project_id = project_id
        self.location = location
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=location)
        
        # Get the endpoint for Model Garden deployment
        if "/" in endpoint_id:
            # Full resource name provided
            self.endpoint = aiplatform.Endpoint(endpoint_id)
        else:
            # Just the endpoint ID provided - construct full resource name
            endpoint_resource_name = f"projects/{project_id}/locations/{location}/endpoints/{endpoint_id}"
            self.endpoint = aiplatform.Endpoint(endpoint_resource_name)
        
        self.model_id = f"gemma3-{endpoint_id}"  # For logging purposes
    
    def generate(self, image_path: Path, prompt: str, seed: int) -> str:
        """Generate response using Gemma 3 with the given image and prompt.
        
        Args:
            image_path: Path to the image file
            prompt: Text prompt to send to the model
            seed: Random seed (note: may not be supported by all endpoints)
        
        Returns:
            Generated text response
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Use Prediction API for Model Garden / vLLM deployments
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        
        # vLLM/Model Garden Chat Completions format
        # NOTE: Image must come BEFORE text in content array
        instance = {
            "@requestFormat": "chatCompletions",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "max_tokens": 512,
            "temperature": self.temperature,
        }
        
        try:
            # No separate parameters dict - everything is in the instance
            response = self.endpoint.predict(instances=[instance])
            
            # Handle response.predictions being a dict or list
            predictions = response.predictions
            
            if predictions:
                # Try to get the first prediction
                if isinstance(predictions, list) and len(predictions) > 0:
                    pred = predictions[0]
                elif isinstance(predictions, dict):
                    # If it's a dict, try common keys
                    if "predictions" in predictions:
                        pred = predictions["predictions"]
                        if isinstance(pred, list) and len(pred) > 0:
                            pred = pred[0]
                    else:
                        # The dict itself might be the prediction
                        pred = predictions
                else:
                    pred = predictions
                
                # Handle different response formats
                if isinstance(pred, dict):
                    if "choices" in pred and pred["choices"]:
                        # Standard OpenAI format
                        choice = pred["choices"][0]
                        if isinstance(choice, dict) and "message" in choice:
                            return choice["message"]["content"]
                        return str(choice)
                    
                    # Fallback: try other common keys
                    for key in ["generated_text", "text", "output", "response", "content"]:
                        if key in pred:
                            return str(pred[key])
                    
                    # Return whole dict as string
                    return str(pred)
                
                # Direct string response
                return str(pred)
            
            return "No prediction returned from model."
            
        except Exception as e:
            print(f"Error with Vertex Prediction API: {e}")
            import traceback
            traceback.print_exc()
            raise
