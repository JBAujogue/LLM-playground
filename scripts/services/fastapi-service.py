"""
Created By  : JBAujogue
Created Date: 2022/01/21
Description : Wrapper converting a LLM into a FastAPI app object.
References  :
    - https://luis-sena.medium.com/how-to-optimize-fastapi-for-ml-model-serving-6f75fb9e040d
    - https://blog.stackademic.com/streaming-llm-responses-using-fastapi-deb575554397
    - https://blog.stackademic.com/streaming-responses-from-llm-using-langchain-fastapi-329f588d3b40
    
Run it in the activated env with :
```
uvicorn scripts.services.fastapi-service:app --root-path . --host 0.0.0.0 --port 8000
```
"""

from typing import Dict, List, Union

# service
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware

# nlp
import torch
from transformers import pipeline



def create_service(llm):
    """
    Creates FastAPI app serving a Named Entity Recognition model
    as a POST method at the '/predict' route.

    Parameters:
        ner_model: an object with a 'predict' method that accepts
        List[str] inputs and returns a pandas.DataFrame as output.

    Returns:
        app: FastAPI app.
    """
    app = FastAPI(llm = llm)

    @app.get("/")
    async def root() -> Dict[str, str]:
        """
        Root route, to check for service sanity.

        Returns:
            Dict[str, str]: Returns a simple 'healthy' message.
        """
        return {"message": "Healthy"}

    @app.post("/predict")
    def predict(body: Dict[str, Union[List[List[dict]], Dict]] = Body(...)) -> List[str]:
        """
        Runs inference of a ML model on body parameters.

        Parameters:
            body: Dict[str, List[str]].
            the json content of the POST request sended to the route.
            Must be a dictionary having an 'chat_messages' key.

        Returns:
            List[Dict]: 
        """
        chat_messages = body["chat_messages"]
        generation_params = body["generation_params"] 
        
        messages = [
            app.extra["llm"].tokenizer.apply_chat_template(
                m, tokenize = False, add_generation_prompt = True,
            )
            for m in chat_messages
        ]
        answers = app.extra["llm"](messages, **generation_params)
        answers = [
            a[0]["generated_text"].split('<|assistant|>')[-1].strip() 
            for a in answers
        ]
        return answers

    return app


# GPU
app = create_service(
    pipeline(
        task = 'text-generation', 
        model = "TheBloke/zephyr-7B-beta-GPTQ", 
        device_map = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    )
)


