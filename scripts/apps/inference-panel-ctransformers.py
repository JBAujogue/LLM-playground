"""
Demonstrates how to use the ChatInterface widget to create a chatbot using
a list of LLMs.
References:
- https://sophiamyang.medium.com/building-ai-chatbots-with-mistral-and-llama2-9c0f5abc296c
- https://blog.holoviz.org/posts/mixtral/
- https://huggingface.co/blog/sophiamyang/tweak-mpl-chat

Run it with
```
panel serve scripts/apps/inference-panel-ctransformers.py --show --args scripts/apps/conf/inference-panel-ctransformers.yaml
```
"""
import yaml
import panel as pn
import argparse


def load_ctransformers_model(model_config):
    from ctransformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(**model_config)


async def run_ctransformers_inference(llm, prompt_config, generation_config):
    prompt = prompt_config['template'].format(**prompt_config['fields'])
    return llm(prompt, **generation_config)


async def callback(
    query: str, user: str, instance: pn.chat.ChatInterface,
    ):
    for model_name in pn.state.cache['models']:
        # set model
        model_args, model = pn.state.cache['models'][model_name]
        
        # set prompt config
        prompt_config = dict(model_args['prompt_config'])
        prompt_config['fields'] = prompt_config['fields'] or dict()
        prompt_config['fields'] |= {'user': query}
        
        # set generation config
        generation_config = model_args['generation_config']
        
        # request response as coroutine
        response = await run_ctransformers_inference(
            model, prompt_config, generation_config,
        )
        # stream response
        message = None
        for chunk in response:
            message = instance.stream(
                chunk, user = model_name, message = message,
            )


# ------------- chat interface ---------------
def parse_model_config_from_cli():
    # parse config filepath, supposed to appear first after --args 
    config_filepath = argparse.ArgumentParser().parse_known_args()[-1][0]

    # parse config file
    with open(config_filepath, "r") as f:
        return yaml.safe_load(f)


def load_model_list_into_panel_cache():
    pn.state.cache['models'] = dict()

    # parse model config list from cli
    model_config_list = parse_model_config_from_cli()

    # load each model listed in config
    for model in model_config_list:
        (model_name, model_args), = model.items()

        # store loaded model into panel cache
        pn.state.cache['models'][model_name] = [
            model_args, load_ctransformers_model(model_args['model_config']),
        ]
    return



pn.extension()
load_model_list_into_panel_cache()

chat_interface = pn.chat.ChatInterface(
    callback = callback,
    callback_exception = 'verbose',
)
chat_interface.send(
    "Send a message to get a reply from LLMs!", 
    user = "System", 
    respond = False,
)
chat_interface.servable()
