"""
Demonstrates how to use the ChatInterface widget to create a chatbot using
a list of LLMs.
References:
- https://sophiamyang.medium.com/building-ai-chatbots-with-mistral-and-llama2-9c0f5abc296c
- https://blog.holoviz.org/posts/mixtral/
- https://huggingface.co/blog/sophiamyang/tweak-mpl-chat

Run it with
```
panel serve scripts/apps/inference-panel.py --autoreload --show --args scripts/apps/conf/inference.yaml
```
"""
import panel as pn
import argparse
from omegaconf import OmegaConf




# ------------- loaders ---------------
def load_ctransformers_model(model_config):
    from ctransformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(**model_config)


def load_langchain_ctransformers_model(model_config):
    '''
    see source code at
    https://api.python.langchain.com/en/latest/llms/langchain_community.llms.ctransformers.CTransformers.html?highlight=ctransformers
    '''
    from langchain.llms import CTransformers
    return CTransformers(**model_config)


def load_huggingface_pipeline(model_config):
    from transformers import pipeline
    return pipeline(**model_config)


def load_huggingface_model(model_config):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(**model_config)
    model = AutoModelForCausalLM.from_pretrained(**model_config)
    return (tokenizer, model)



loader_mapping = {
    'huggingface': load_huggingface_model,
    'huggingface-pipeline': load_huggingface_pipeline,
    'ctransformers': load_ctransformers_model,
    'langchain-ctransformers': load_langchain_ctransformers_model,
}




# ------------- inference wrappers ---------------
async def run_ctransformers_inference(llm, prompt_config, generation_config):
    prompt = prompt_config['template'].format(**prompt_config['fields'])
    return llm(prompt, **generation_config)


async def run_langchain_ctransformers_inference(llm, prompt_config, generation_config):
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
            
    # predefine template wrapper
    prompt = PromptTemplate(**prompt_config)
    # put into chain
    llm = LLMChain(llm = llm, prompt = prompt)
    return llm.predict(**prompt_config['fields'])


async def run_huggingface_pipeline_inference(llm, prompt_config, generation_config):
    if 'stream' in generation_config:
        if generation_config['stream']:
            from transformers import TextStreamer
            generation_config['streamer'] = TextStreamer(
                llm.tokenizer, skip_prompt = True, skip_special_tokens = True,
            )
        generation_config.pop('stream')
    
    message = [
        dict(role = k, content = v)
        for k, v in prompt_config['fields'].items()
    ]
    message = llm.tokenizer.apply_chat_template(
        message, tokenize = False, add_generation_prompt = False
    )
    return llm(message, **generation_config)[0]['generated_text']


async def run_huggingface_inference(llm, prompt_config, generation_config):
    tokenizer, model = llm
    return



inference_mapping = {
    'huggingface': run_huggingface_inference,
    'huggingface-pipeline': run_huggingface_pipeline_inference,
    'ctransformers': run_ctransformers_inference,
    'langchain-ctransformers': run_langchain_ctransformers_inference,
}



# ------------- callback ---------------
async def callback(
    query: str, user: str, instance: pn.chat.ChatInterface,
    ):
    for model in pn.state.cache['model_config_list']:
        (model_name, model_args), = model.items()
        engine = model_args['engine']
        
        # select model from app state cache
        if model_name not in pn.state.cache:
            loader = loader_mapping[engine]
            pn.state.cache[model_name] = loader(model_args['model_config'])
            
        llm = pn.state.cache[model_name]
        
        # set prompt and generation config
        prompt_config = model_args['prompt_config']
        prompt_config = prompt_config or {}
        prompt_config = {k: (v or {}) for k, v in prompt_config.items()}
        prompt_config['fields'] |= {'user': query}
        
        generation_config = model_args['generation_config']
        generation_config = generation_config or {}
        
        # request response
        response = await inference_mapping[engine](llm, prompt_config, generation_config)

       # send response to app
        stream = model_args['panel_config']['stream']
        if stream:
            # stream response
            message = None
            for chunk in response:
                message = instance.stream(
                    chunk, user = model_name, message = message,
                )
        else:
            # send response
            instance.send(response, user = model_name, respond = False)



# ------------- chat interface ---------------
def parse_model_config_from_cli():
    # parse config filepath, supposed to appear first after --args 
    parser = argparse.ArgumentParser()
    print(parser.parse_known_args())
    model_config_filepath = parser.parse_known_args()[-1][0]
    
    # load config
    return OmegaConf.to_object(OmegaConf.load(model_config_filepath))


model_list = [
    # 'zephyr-7b-gguf-ctransformers-cpu',
    # 'mistral-7b-instruct-v0.2-gguf-ctransformers-cpu',
    # 'zephyr-7b-gguf-langchain-ctransformers-cpu',
    # 'mistral-7b-instruct-v0.2-gptq-huggingface-pipeline-cuda',
]

if 'model_config_list' not in pn.state.cache:
    model_config_list = parse_model_config_from_cli()
    model_config_list = [
        cfg for cfg in model_config_list
        if list(cfg)[0] in model_list
    ]
    pn.state.cache['model_config_list'] = model_config_list


pn.extension()
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
