

"""
Demonstrates how to use the ChatInterface widget to create a chatbot using
a list of LLMs. Taking inspiration from 
https://sophiamyang.medium.com/building-ai-chatbots-with-mistral-and-llama2-9c0f5abc296c
Run it with
```
panel serve apps/simple-chat-panel.py
```
"""

import panel as pn
from langchain.chains import LLMChain
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate

pn.extension()

engine = 'ctransformers'
model_dict = {
    'zephyr-7b-gptq': {
        "model": 'TheBloke/zephyr-7B-beta-GPTQ',
    },
    'zephyr-7b-gguf': {
        "model": 'TheBloke/zephyr-7B-beta-GGUF',
    },
}
generation_config = dict(max_new_tokens = 256, temperature = 0.5)
llm_chains = {}
template = """<s>[INST] You are a friendly chat bot who's willing to help answer the
user:
{user_input} [/INST] </s>
"""


def create_model(engine: str, model_config: dict, generation_config: dict):
    if engine == 'ctransformers':
        return CTransformers(**model_config, config = generation_config)
    else:
        raise NotImplementedError(
            '''
            accepted engine are: 
            - 'ctransformers'
            '''
        )

async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    '''
    '''
    for model, model_config in model_dict.items():
        if model not in llm_chains:
            instance.placeholder_text = (f"Downloading {model}")
            llm = create_model(engine, model_config, generation_config)
            prompt = PromptTemplate(template = template, input_variables=["user_input"])
            llm_chain = LLMChain(prompt = prompt, llm = llm)
            llm_chains[model] = llm_chain
        instance.send(
            await llm_chains[model].apredict(user_input = contents),
            user = model.title(),
            respond = False,
        )


chat_interface = pn.chat.ChatInterface(callback = callback, placeholder_threshold = 0.1)
chat_interface.send(
    "Send a message to get a reply from Zephyr-7B!",
    user = "System",
    respond=False,
)
chat_interface.servable()