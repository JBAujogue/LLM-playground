# Introduction 
This project contains experiments on GenAI.


# Getting Started
## Installation process
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
            <td style="font-size:20px">Paper</td>
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
            <td> </td>
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
            <td>LangChain</td>
            <td> </td>
            <td> </td>
            <td> </td>
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
            <td> </td>
        </tr>
        <tr>
            <td>EmbedChain</td>
            <td> </td>
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
                <a href="https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard">Open LLM Huggingface leaderboard</a>
            </td>
        </tr>
        <tr><th colspan=4; style="height:100%; width:100%; text-align:center; font-size:30px"> * </th></tr>
    </tbody>
    <! ----------------- 2. Quantization ------------------>
    <thead>
        <tr>
            <th colspan=4; style="text-align:center; font-size:30px">2. Quantization</th>
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
        <tr><th colspan=4; style="height:100%; width:100%; text-align:center; font-size:30px"> * </th></tr>
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




