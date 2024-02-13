# Introduction 
This project contains experiments on GenAI.


# Getting Started
## Local python setup
This project uses python `3.11` as core interpreter, and poetry `1.6.1` as dependency manager.
1) Create a new conda environment with
```
conda env create -f environment.yml
```

2) Activate the environment with
```
conda activate llm-playground
```

3) Move to the project directory, and install the project dependencies with
```
poetry install
```

4) Launch a jupyter server with
```
jupyter notebook
```

## Docker & WSL
This project was developed on a Windows 11 os, while some components require a linux os and are thus running inside a containerized environment backed by WSL.

1) Install WSL and a Linux distribution following the [Microsoft official doc](https://learn.microsoft.com/en-us/windows/wsl/install).

2) Install [Docker](https://docs.docker.com/desktop/install/windows-install/) or [Podman]().

Terminate the running linux kernel:
```
(kill one kernel) wsl -t <distro-name>
(kill all kernel) wsl --shutdown
```

# Learning plan




<table style="height:100%; width:100%; text-align:center;">
    <thead>
        <tr>
            <th colspan=4; style="height:100%; width:100%; text-align:center; font-size:30px">1. Inference</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="font-size:20px">Method</td>
            <td style="font-size:20px">Documentation</td>
            <td style="font-size:20px">Examples</td>
            <td style="font-size:20px">Comment</td>
        </tr>
        <tr>
            <td>Huggingface transformers</td>
            <td> </td>
            <td> </td>
            <td> </td>
        </tr>
        <tr>
            <td>ctransformers</td>
            <td><a href="https://github.com/marella/ctransformers">Github</a></td>
            <td> </td>
            <td>CPU-only, Unmaintained</td>
        </tr>
        <tr>
            <td>ollama</td>
            <td><a href="https://github.com/jmorganca/ollama">Github</a></td>
            <td><a href="https://blog.llamaindex.ai/running-mixtral-8x7-locally-with-llamaindex-e6cebeabe0ab">ollama for Mixtral</a></td>
            <td> </td>
        </tr>
        <tr>
            <td>OpenLLM</td>
            <td><a href="https://github.com/bentoml/OpenLLM">Github</a></td>
            <td> </td>
            <td> </td>
        </tr>
        <tr>
            <td>PrivateGPT</td>
            <td><a href="https://github.com/imartinez/privateGPT">Github</a></td>
            <td> </td>
            <td> </td>
        </tr>
        <tr>
            <td>DeepSparse</td>
            <td><a href="https://github.com/neuralmagic/deepsparse">Github</a></td>
            <td> </td>
            <td> </td>
        </tr>
        <tr>
            <td>Llama-cpp-python</td>
            <td><a href="https://github.com/abetlen/llama-cpp-python">Github</a></td>
            <td> </td>
            <td> </td>
        </tr>
        <tr>
            <td>vLLM</td>
            <td>
                <a href="https://github.com/vllm-project/vllm">Github</a>,
                <a href="https://www.anyscale.com/blog/continuous-batching-llm-inference">Inference speed blog post</a>,
                <a href="https://arxiv.org/pdf/2309.06180.pdf">Article</a>
            </td>
            <td>
                <a href="https://docs.vllm.ai/en/latest/getting_started/quickstart.html">Official quickstart</a>, 
                <a href="https://github.com/vllm-project/vllm/tree/main/examples">Official list of examples</a>,
                <a href="https://betterprogramming.pub/superfast-llm-text-generation-with-vllm-on-windows-11-4a6617d4e0b3">Run in WSL</a>
            </td>
            <td>Service, Linux-only</td>
        </tr>
        <tr>
            <td>TGI: Text Generation Inference</td>
            <td>
                <a href="https://github.com/huggingface/text-generation-inference">Github</a>
                <a href="https://huggingface.co/docs/text-generation-inference/index">HF page</a>
            </td>
            <td> 
                <a href="https://github.com/yjg30737/windows-text-generation-inference-example">Run with WSL & Docker</a>
            </td>
            <td>Service, Linux-only</td>
        </tr>
        <tr>
            <td>Triton Inference Server</td>
            <td>
                <a href="https://github.com/triton-inference-server/pytriton/tree/main">Github</a>
                <a href="https://github.com/triton-inference-server/pytriton/tree/main">pytriton Github</a>
            </td>
            <td> </td>
            <td>Service, Linux-only</td>
        </tr>
        <tr>
            <td>LangChain</td>
            <td> </td>
            <td> </td>
            <td>Framework</td>
        </tr>
        <tr>
            <td>Llama-index</td>
            <td>
                <a href="https://github.com/run-llama/llama_index">Github</a>,
                <a href="https://docs.llamaindex.ai/en/stable/">Documentation</a>
            </td>
            <td>
                <a href="https://github.com/run-llama/llama_index/tree/main/docs/examples/llm">Official list of notebooks</a>
            </td>
            <td>Framework</td>
        </tr>
        <tr>
            <td>EmbedChain</td>
            <td> </td>
            <td> </td>
            <td> </td>
        </tr>
        <tr>
            <td>Jan (product)</td>
            <td> 
                <a href="https://github.com/janhq/jan/tree/dev">
                Github
                </a>
            </td>
            <td> </td>
            <td> </td>
        </tr>
        <tr>
            <td colspan=4; style="font-size:20px">
                - Further readings -
            </td>
        </tr>
        <tr>
            <td colspan=4;>
                <a href="https://docs.llamaindex.ai/en/stable/module_guides/models/llms.html#modules">List of LLM frameworks</a>,
                <a href="https://github.com/aws-samples/amazon-sagemaker-generativeai">AWS GenAI tutorials</a>,
                <a href="https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard">Open LLM Huggingface leaderboard</a>,
                <a href="https://github.com/hamelsmu/llama-inference">hamelsmu llama-inference</a>,
            </td>
        </tr>
        <tr><th colspan=4; style="height:100%; width:100%; text-align:center; font-size:30px"> * </th></tr>
        <tr><th colspan=4; style="height:100%; width:100%; text-align:center; font-size:30px"></th></tr>
    </tbody>
    <! ----------------- 2. Compression, Quantization ------------------>
    <thead>
        <tr>
            <th colspan=4; style="text-align:center; font-size:30px">2. Compression, Quantization</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="font-size:20px">Method</td>
            <td style="font-size:20px">Documentation</td>
            <td style="font-size:20px">Examples</td>
            <td style="font-size:20px">Paper</td>
        </tr>
        <tr>
            <td>SparseML</td>
            <td> 
                <a href="https://github.com/neuralmagic/sparseml">
                    <div style="text-align: center;">Github</div>
                </a>
            </td>
            <td> </td>            
            <td> </td>
        </tr>
        <tr>
            <td>BitsAndBytes</td>
            <td>
                <a href="https://huggingface.co/docs/transformers/quantization#bitsandbytes">
                    <div style="text-align: center;">HF docs</div>
                </a>
            </td>
            <td>
                <a href="https://huggingface.co/docs/transformers/quantization#bitsandbytes">
                    <div style="text-align: center;">HF docs</div>
                </a>
            </td>            
            <td>
                <a href="https://arxiv.org/pdf/2208.07339.pdf">
                    <div style="text-align: center;">2208</div>
                </a>
            </td>
        </tr>
        <tr>
            <td>GPTQ</td>
            <td>
                <a href="https://huggingface.co/blog/gptq-integration">
                    <div style="text-align: center;">HF blog</div>
                </a>
            </td>
            <td>
                <a href="https://github.com/PanQiWei/AutoGPTQ/tree/main/docs/tutorial">
                    <div style="text-align: center;">Official repo notebooks</div>
                </a>
            </td>            
            <td>
                <a href="https://arxiv.org/pdf/2210.17323.pdf">
                    <div style="text-align: center;">2210</div>
                </a>
            </td>
        </tr>
        <tr>
            <td>AWQ: Activation-aware Weight Quantization</td>
            <td>
                <a href="https://huggingface.co/docs/transformers/quantization#awq">
                    <div style="text-align: center;">HF docs</div>
                </a>
            </td>
            <td>
                <a href="https://colab.research.google.com/drive/1HzZH89yAXJaZgwJDhQj9LqSBux932BvY">
                    <div style="text-align: center;">notebook</div>
                </a>
            </td>            
            <td>
                <a href="https://arxiv.org/pdf/2306.00978.pdf">
                    <div style="text-align: center;">2306</div>
                </a>
            </td>
        </tr>
        <tr>
            <td>SqueezeLLM</td>
            <td> </td>
            <td> </td>            
            <td>
                <a href="https://arxiv.org/pdf/2306.07629.pdf">
                    <div style="text-align: center;">2306</div>
                </a>
            </td>
        </tr>
        <tr>
            <td>EXL2</td>
            <td>
                <a href="https://github.com/turboderp/exllamav2">Github</a>
            </td>
            <td>
                <a href="https://towardsdatascience.com/exllamav2-the-fastest-library-to-run-llms-32aeda294d26">Blog post</a>
            </td>
            <td></td>
        </tr>
        <tr>
            <td>HQQ: Half-Quadratic Quantization</td>
            <td>
                <a href="https://github.com/mobiusml/hqq">Github</a>
            </td>
            <td>
                <a href="https://towardsdatascience.com/run-mixtral-8x7b-on-consumer-hardware-with-expert-offloading-bd3ada394688">HQQ for Mixtral</a>
            </td>
            <td></td>
        </tr>        
        <tr><th colspan=4; style="height:100%; width:100%; text-align:center; font-size:30px"> * </th></tr>
        <tr><th colspan=4; style="height:100%; width:100%; text-align:center; font-size:30px"></th></tr>
    </tbody>
    <! ----------------- 3. Evaluation ------------------>
    <thead>
        <tr>
            <th colspan=4; style="height:100%; width:100%; text-align:center; font-size:30px">3. Evaluation</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="font-size:20px">Method</td>
            <td style="font-size:20px">Documentation</td>
            <td style="font-size:20px">Examples</td>
            <td style="font-size:20px">Github</td>
        </tr>
        <tr>
            <td >LLM-autoeval</td>
            <td > </td>
            <td > </td>
            <td href="https://github.com/mlabonne/llm-autoeval">repo</td>
        </tr>
        <tr><th colspan=4; style="height:100%; width:100%; text-align:center; font-size:30px"> * </th></tr>
        <tr><th colspan=4; style="height:100%; width:100%; text-align:center; font-size:30px"></th></tr>
    </tbody>
    <! ----------------- 4. Prompt Engineering ------------------>
    <thead>
        <tr>
            <th colspan=4; style="height:100%; width:100%; text-align:center; font-size:30px">4. Prompt Engineering</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="font-size:20px">Method</td>
            <td style="font-size:20px">Documentation</td>
            <td style="font-size:20px">Examples</td>
            <td style="font-size:20px">Paper</td>
        </tr>
        <tr>
            <td>Chain of thoughts</td>
        </tr>
        <tr>
            <td>Tree of thoughts</td>
        </tr>
        <tr>
            <td>Graph of thoughts</td>
        </tr>
        <tr>
            <td>Prompt injection</td>
        </tr>
        <tr><th colspan=4; style="height:100%; width:100%; text-align:center; font-size:30px"> * </th></tr>
        <tr><th colspan=4; style="height:100%; width:100%; text-align:center; font-size:30px"></th></tr>
    </tbody>
    <! ----------------- 5. Data Ingestion ------------------>
    <thead>
        <tr>
            <th colspan=4; style="height:100%; width:100%; text-align:center; font-size:30px">5. Data Ingestion</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="font-size:20px">Method</td>
            <td style="font-size:20px">Documentation</td>
            <td style="font-size:20px">Examples</td>
            <td style="font-size:20px">Paper</td>
        </tr>
        <tr>
            <td>Automatic Data Selection in Instruction Tuning</td>
            <td> </td>
            <td> </td>
            <td>
                <a href="https://arxiv.org/pdf/2312.15685.pdf">
                    <div style="text-align: center;">2312</div>
                </a>
            </td>
        </tr>
        <tr><th colspan=4; style="height:100%; width:100%; text-align:center; font-size:30px"> * </th></tr>
        <tr><th colspan=4; style="height:100%; width:100%; text-align:center; font-size:30px"></th></tr>
    </tbody>
    <! ----------------- 6. Retrieval-Augmented Generation ------------------>
    <thead>
        <tr>
            <th colspan=4; style="height:100%; width:100%; text-align:center; font-size:30px">6. Retrieval-Augmented Generation</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="font-size:20px">Method</td>
            <td style="font-size:20px">Documentation</td>
            <td style="font-size:20px">Examples</td>
            <td style="font-size:20px">Paper</td>
        </tr>
        <tr>
            <td>RAG</td>
            <td> </td>
            <td>
                <a href="https://blog.llamaindex.ai/a-cheat-sheet-and-some-recipes-for-building-advanced-rag-803a9d94c41b">llama-index blog</a>,
                <a href="https://docs.llamaindex.ai/en/stable/getting_started/concepts.html#retrieval-augmented-generation-rag">llama-index documentation</a>
            </td>
            <td><a href="https://arxiv.org/pdf/2312.10997.pdf">2312</a></td>
        </tr>
        <tr>
            <td>self-RAG</td>
        </tr>
        <tr><th colspan=4; style="height:100%; width:100%; text-align:center; font-size:30px"> * </th></tr>
        <tr><th colspan=4; style="height:100%; width:100%; text-align:center; font-size:30px"></th></tr>
    </tbody>
    <! ----------------- 7. Finetuning ------------------>
    <thead>
        <tr>
            <th colspan=4; style="height:100%; width:100%; text-align:center; font-size:30px">7. Finetuning</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="font-size:20px">Method</td>
            <td style="font-size:20px">Documentation</td>
            <td style="font-size:20px">Examples</td>
            <td style="font-size:20px">Paper</td>
        </tr>
        <tr>
            <td>PEFT: Parameter-Efficient FineTuning</td>
            <td> </td>
            <td> </td>
            <td> </td>
        </tr>
        <tr>
            <td>C-RLFT: Conditioned-Reinforcement Learning Fine-Tuning</td>
            <td> </td>
            <td> </td>
            <td> </td>
        </tr>
        <tr>
            <td>LoRA: Low Ranking Adaptation</td>
            <td> </td>
            <td> </td>
            <td> </td>
        </tr>
        <tr>
            <td>QLoRA: Quantized Low Ranking Adaptation</td>
            <td> </td>
            <td> </td>
            <td> </td>
        </tr>
       <tr>
            <td>DPO: Direct Preference Optimization</td>
            <td> </td>
            <td> </td>
            <td> </td>
        </tr>
        <tr>
            <td>SPIN: Self-Play Finetuning</td>
            <td> </td>
            <td> </td>
            <td>
                <a href="https://arxiv.org/pdf/2401.01335.pdf">
                    <div style="text-align: center;">2401</div>
                </a>
            </td>
        </tr>
        <tr>
            <td>ASTRAIOS: Parameter-Efficient Instruction Tuning</td>
            <td> </td>
            <td> </td>
            <td>
                <a href="https://arxiv.org/pdf/2401.00788.pdf">
                    <div style="text-align: center;">2401</div>
                </a>
            </td>
        </tr>
        <tr>
            <td>LLAMA-pro: Progressive Learning of LLMs</td>
            <td> </td>
            <td> </td>
            <td> </td>
        </tr>
        <! ----------------- 7.2 Frameworks ------------------>
        <tr><th colspan=4; style="height:100%; width:100%; text-align:center; font-size:30px"> </th></tr>
        <tr>
            <td style="font-size:20px">Framework</td>
            <td style="font-size:20px">Documentation</td>
            <td style="font-size:20px">Examples</td>
            <td style="font-size:20px">Paper</td>
        </tr>
        <tr>
            <td>HF autotrain-advanced</td>
            <td>
                <a href="https://github.com/huggingface/autotrain-advanced">
                Github
                </a>
            </td>
            <td> </td>
            <td> </td>
        </tr>
        <tr><th colspan=4; style="height:100%; width:100%; text-align:center; font-size:30px"> * </th></tr>
        <tr><th colspan=4; style="height:100%; width:100%; text-align:center; font-size:30px"></th></tr>
    </tbody>
    <! ----------------- 8. Model aggregation ------------------>
    <thead>
        <tr>
            <th colspan=4; style="height:100%; width:100%; text-align:center; font-size:30px">8. Model aggregation</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="font-size:20px">Method</td>
            <td style="font-size:20px">Documentation</td>
            <td style="font-size:20px">Examples</td>
            <td style="font-size:20px">Paper</td>
        </tr>
        <tr>
            <td>MoE: Mixture of Experts</td>
            <td> </td>
            <td> </td>
            <td>
                <a href="https://arxiv.org/pdf/2209.01667.pdf">
                    <div style="text-align: center;">2209</div>
                </a>
            </td>
        </tr>
        <tr>
            <td>Model merging</td>
            <td>  
                <a href="https://huggingface.co/blog/mlabonne/merge-models?trk=feed_main-feed-card_feed-article-content">HF blog</a>,
                <a href="https://huggingface.co/collections/osanseviero/model-merging-65097893623330a3a51ead66">Model merging bibliography</a>
            </td>
            <td> </td>
            <td> </td>
        </tr>
        <tr><th colspan=4; style="height:100%; width:100%; text-align:center; font-size:30px"> * </th></tr>
        <tr><th colspan=4; style="height:100%; width:100%; text-align:center; font-size:30px"></th></tr>
    </tbody>
    <! ----------------- 9. Agents ------------------>
    <thead>
        <tr>
            <th colspan=4; style="height:100%; width:100%; text-align:center; font-size:30px">9. Agents</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="font-size:20px">Method</td>
            <td style="font-size:20px">Documentation</td>
            <td style="font-size:20px">Examples</td>
            <td style="font-size:20px">Paper</td>
        </tr>
    </tbody>
</table>


## Notables LLMs
- OpenChat
- OpenHermes
- mistral-OpenOrca
- zephyr-beta
- mistral
- llama2



## References
- [llm-course](https://github.com/mlabonne/llm-course/tree/main) Github repository
- [langchain](https://github.com/langchain-ai/langchain/tree/master) Github repository
- [llm-autoeval](https://github.com/mlabonne/llm-autoeval) Github repository
-[LLMs-from-scratch]()



<table style="height:100%; width:100%; text-align:center;">
    <thead>
        <tr>
            <th colspan=4; style="height:100%; width:100%; text-align:center; font-size:30px">Asset references</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="font-size:20px">Inference</td>
            <td style="font-size:20px">Leverage external source</td>
            <td style="font-size:20px">Multi-turn interaction</td>
            <td style="font-size:20px">Reasoning & intermediate steps</td>
            <td style="font-size:20px">Agents</td>
        </tr>
        <tr>
            <td>
                <a href="https://sophiamyang.medium.com/building-ai-chatbots-with-mistral-and-llama2-9c0f5abc296c">
                ✅ Building AI Chatbots with Mistral and Llama2
                </a>
                <a href="https://betterprogramming.pub/frameworks-for-serving-llms-60b7f7b23407">
                🔲 7 Frameworks for Serving LLMs
                </a>
            </td>
            <td>
                <a href="https://blog.llamaindex.ai/a-cheat-sheet-and-some-recipes-for-building-advanced-rag-803a9d94c41b">
                🔲 A Cheat Sheet and Some Recipes For Building Advanced RAG
                </a>
                <a href="https://towardsdatascience.com/why-are-advanced-rag-methods-crucial-for-the-future-of-ai-462e0dc5a208">
                🔲 Why Are Advanced RAG Methods Crucial for the Future of AI?
                </a>
            </td>
            <td> </td>
            <td> </td>
            <td>
                <a href="https://github.com/run-llama/llama-lab">
                🔲 Llama-lab
                </a>
            </td>
        </tr>
        <tr>
    </tbody>
</table>


https://python.plainenglish.io/intruct-fine-tuning-mistral-7b-model-with-your-custom-data-7eb22921a483
