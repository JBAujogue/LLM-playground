

"""
Demonstrates how to use the ChatInterface widget to create a chatbot using
a list of LLMs. Taking inspiration from 
https://sophiamyang.medium.com/building-ai-chatbots-with-mistral-and-llama2-9c0f5abc296c
Run it with
```
panel serve scripts/apps/simple-chat-panel.py
```
"""

import panel as pn
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate




# ------------- ctransformers ---------------
MODEL_ARGUMENTS = {
    # "zephyr-7b-gptq-cuda": {
    #     "args": ["TheBloke/zephyr-7B-beta-GPTQ"],
    #     "kwargs": dict(
    #         model_type = "gptq",
    #     ),
    # },
    "zephyr-7b-gguf-cpu": dict(
        args = ["TheBloke/zephyr-7B-beta-GGUF"],
        kwargs = dict(
            model_file = "zephyr-7b-beta.Q5_K_M.gguf",
        ),
    ),
}


def load_ctransformers_model(model_args):
    from ctransformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
                *model_args["args"],
                **model_args["kwargs"],
            )


async def callback_ctransformers(
    query: str, user: str, instance: pn.chat.ChatInterface,
    ):
    for model, model_args in MODEL_ARGUMENTS.items():
        # load model in app state cache
        if model not in pn.state.cache:
            pn.state.cache[model] = load_ctransformers_model(model_args)
        # select model
        llm = pn.state.cache[model]
        
        # wrap content into template on-the-fly
        system = 'Answer the user query, in french, in minimalistic style.'
        prompt = f'''<|system|>\n{system}</s>\n<|user|>\n{query}</s>\n<|assistant|>'''
        
        # request response
        response = llm(
            prompt, **pn.state.cache['generation_config'],
        )
        # stream response on chat interface
        message = None
        for chunk in response:
            message = instance.stream(
                chunk, user = model.title(), message = message,
            )




# ------------- langchain wrapper for ctransformers ---------------
langchain_ctransformers_model_args = {
    "zephyr-7b-gguf-cpu": dict(
        model = "TheBloke/zephyr-7B-beta-GGUF",
        model_file = "zephyr-7b-beta.Q5_K_M.gguf",
    ),
}


def load_langchain_ctransformers_model(model_args, config: dict = {}):
    '''
    see source code at
    https://api.python.langchain.com/en/latest/llms/langchain_community.llms.ctransformers.CTransformers.html?highlight=ctransformers
    '''
    from langchain.llms import CTransformers
    return CTransformers(
        **model_args, config = config,
    )


async def callback_langchain_ctransformers(
    contents: str, user: str, instance: pn.chat.ChatInterface,
    ):
    for model, model_args in langchain_ctransformers_model_args.items():
        # load model in app state cache
        if model not in pn.state.cache:
            llm = load_langchain_ctransformers_model(
                model_args, pn.state.cache['generation_config'],
            )
            
            # predefine template wrapper
            template = '''<|system|>\n{system}</s>\n<|user|>\n{query}</s>\n<|assistant|>'''
            prompt = PromptTemplate(
                template = template, input_variables = ['system', 'query'],
            )
            # put model into cache
            pn.state.cache[model] = LLMChain(llm = llm, prompt = prompt)
        
        # select model
        llm = pn.state.cache[model]
        
        # define system profile
        system = 'Answer the user query, in french, in minimalistic style.'

        # request and send response on chat interface.
        # as of 01/2024, langchain + ctransformers doesn't 
        # support streaming, see updates at
        # https://python.langchain.com/docs/integrations/llms/
        instance.send(
            await llm.apredict(system = system, query = contents),
            user = model.title(),
            respond = False,
        )



# ------------- chat interface ---------------
if 'generation_config' not in pn.state.cache:
    pn.state.cache['generation_config'] = dict(
        max_new_tokens = 32,
        stream = True,
    )


pn.extension()

chat_interface = pn.chat.ChatInterface(
    callback = callback_ctransformers,
    callback_exception = 'verbose',
)
chat_interface.send(
    "Send a message to get a reply from both Llama 2 and Mistral (7B)!",
    user = "System",
    respond = False,
)
chat_interface.servable()