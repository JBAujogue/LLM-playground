"""
Demonstrates how to use the ChatInterface widget to create a chatbot using
a list of LLMs.
References:
- https://sophiamyang.medium.com/building-ai-chatbots-with-mistral-and-llama2-9c0f5abc296c
- https://blog.holoviz.org/posts/mixtral/
- https://huggingface.co/blog/sophiamyang/tweak-mpl-chat
- https://sophiamyang.medium.com/how-to-build-your-own-panel-ai-chatbots-ef764f7f114e

Run it with
```
panel serve scripts/apps/inference-panel-langchain.py --show --args scripts/apps/conf/inference-panel-langchain.yaml
```
"""
import yaml
import panel as pn
import argparse

from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate



def load_langchain_ctransformers_model(model_config):
    return CTransformers(**model_config)


def run_langchain_ctransformers_inference(
        llm, prompt_config, callback_handler,
    ):
    # predefine template wrapper
    prompt = PromptTemplate(**prompt_config)

    # put into chain
    llm_chain = LLMChain(llm = llm, prompt = prompt)
    return llm_chain.predict(**prompt_config['fields'], callbacks = [callback_handler])


def callback(
        query: str, user: str, instance: pn.chat.ChatInterface
    ):
    callback_handler = pn.chat.langchain.PanelCallbackHandler(instance)

    for model_name in pn.state.cache['models']:
        # set model
        model_args, model = pn.state.cache['models'][model_name]
        
        # set prompt config
        prompt_config = dict(model_args['prompt_config'])
        prompt_config['fields'] = prompt_config['fields'] or dict()
        prompt_config['fields'] |= {'user': query}

        response = run_langchain_ctransformers_inference(
            model, prompt_config,  callback_handler
        )
    return


def parse_model_config_from_cli():
    # parse config filepath, supposed to appear first after --args 
    config_filepath = argparse.ArgumentParser().parse_known_args()[-1][0]

    # parse config file
    with open(config_filepath, "r") as f:
        return yaml.safe_load(f)


pn.extension()
chat_interface = pn.chat.ChatInterface(
    callback = callback,
    callback_exception = 'verbose',
)

pn.state.cache['models'] = dict()

# parse model config list from cli
model_config_list = parse_model_config_from_cli()

# load each model listed in config
for model_conf in model_config_list:
    (model_name, model_args), = model_conf.items()

    llm = load_langchain_ctransformers_model(model_args['model_config'])

    # store loaded model into panel cache
    pn.state.cache['models'][model_name] = [model_args, llm]

chat_interface.send(
    "Send a message to get a reply from LLMs!", 
    user = "System", 
    respond = False,
)
chat_interface.servable()
