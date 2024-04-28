# Introduction 
This project contains experiments on GenAI.


# Getting Started
## WSL setup
This project was developed on a Windows 11 os, while some components require a linux os and are thus running inside a containerized environment backed by WSL.

Install WSL and create a linux distribution following the [Microsoft official doc](https://learn.microsoft.com/en-us/windows/wsl/install). See also the [wsl basic commands](https://learn.microsoft.com/en-us/windows/wsl/basic-commands).

## Docker setup
1) Install [Docker desktop](https://docs.docker.com/desktop/install/windows-install/) or [Podman]() on the Windows os. 
    - **Note**: All previous versions of Docker Engine and CLI installed directly through Linux distributions must be uninstalled before installing Docker Desktop.
    - Activate the WSL integration in Docker Desktop settings following [docker documentation](https://docs.docker.com/desktop/wsl/#turn-on-docker-desktop-wsl-2)

2) Install `nvidia-container-toolkit` in your linux distro:
    - Launch a linux terminal by running the following command in a cmd (the `--distribution` flag being optional)
      ```bash
      wsl --distribution <distro-name>
      ```
    - Execute the install commands found in [nvidia documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt)
    - Allow docker to use nvidia Container Runtime by executing the commands found in [nvidia documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuration)

3) Additional tips:
    - Terminate the running linux kernel:
      ```bash
      (kill one kernel) wsl -t <distro-name>
      (kill all kernel) wsl --shutdown
      ```

## Database setup
1) Get [qdrant](https://qdrant.tech/documentation/guides/installation/#docker) latest docker image by running (in CMD or bash):
    ```bash
    docker pull qdrant/qdrant
    ```


## Python setup
This project uses python `3.11` as core interpreter, and poetry `1.6.1` as dependency manager.
1) Install miniconda on windows or on [linux](https://dev.to/sfpear/miniconda-in-wsl-3642).
2) Create a new conda environment with
    ```bash
    conda env create -f environment.yml
    ```

3) Activate the environment with
    ```bash
    conda activate llm-playground
    ```

4) Move to the project directory, and install the project dependencies with
    ```bash
    poetry install
    ```

5) Launch a jupyter server with
    ```bash
    jupyter notebook
    ```

# How to use it
## Run Qdrant vector database
We dedicate a docker container with name `qdrant-db` to the vector database backend microservice.
- Create and run new qdrant vector database service:
    ```bash
    wsl -e ./scripts/services/qdrant/qdrant-db.sh
    ```
- Run existing qdrant vector database service:
    ```bash
    docker start qdrant-db
    ```


## Run Text Generation Inference LLM service
We dedicate a docker container with name `tgi-service` to a TGI-based LLM backend microservice.
- Create and run new TGI llm service:
    Launch docker desktop, open a cmd or shell and run
    ```bash
    wsl -e ./scripts/services/tgi/tgi-service.sh
    ```

- Run existing TGI llm service:
    Launch docker desktop, open a cmd or shell and run
    ```bash
    docker start tgi-service
    ```


# Learning plan

<table style="height:100%; width:100%; text-align:center;">
    <! ----------------- 1.1 Frameworks ------------------>
    <thead>
        <tr>
            <th colspan=4; style="height:100%; width:100%; text-align:center; font-size:30px">1. Inference</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="font-size:20px">Framework</td>
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
        <! ----------------- 1.2 Services ------------------>
        <tr><th colspan=4; style="height:100%; width:100%; text-align:center; font-size:30px"> </th></tr>
        <tr>
            <td style="font-size:20px">Service</td>
            <td style="font-size:20px">Documentation</td>
            <td style="font-size:20px">Examples</td>
            <td style="font-size:20px">Comment</td>
        </tr>
        <tr>
            <td>vLLM</td>
            <td>
                <a href="https://github.com/vllm-project/vllm">Github</a>,
                <a href="https://www.anyscale.com/blog/continuous-batching-llm-inference">Inference speed blog post</a>,
                <a href="https://arxiv.org/pdf/2309.06180.pdf">2309</a>
            </td>
            <td>
                <a href="https://docs.vllm.ai/en/latest/getting_started/quickstart.html">Official quickstart</a>, 
                <a href="https://github.com/vllm-project/vllm/tree/main/examples">Official list of examples</a>,
                <a href="https://betterprogramming.pub/superfast-llm-text-generation-with-vllm-on-windows-11-4a6617d4e0b3">Run in WSL</a>
            </td>
            <td>Linux-only</td>
        </tr>
        <tr>
            <td>TGI: Text Generation Inference</td>
            <td>
                <a href="https://github.com/huggingface/text-generation-inference">Github</a>
                <a href="https://huggingface.co/docs/text-generation-inference/index">HF page</a>
            </td>
            <td> 
                <a href="https://github.com/yjg30737/windows-text-generation-inference-example">Run with WSL & Docker</a>,
                <a href="https://kaitchup.substack.com/p/serve-large-language-models-from">Run again with WSL & Docker</a>,
                <a href="https://vilsonrodrigues.medium.com/serving-falcon-models-with-text-generation-inference-tgi-5f32005c663b">External usage</a>,
                <a href="https://huggingface.co/blog/tgi-messages-api">Use with OpenAI / langchain / llama-index client</a>
            </td>
            <td>Linux-only</td>
        </tr>
        <tr>
            <td>Triton Inference Server</td>
            <td>
                <a href="https://github.com/triton-inference-server/pytriton/tree/main">Github</a>,
                <a href="https://github.com/triton-inference-server/pytriton/tree/main">pytriton Github</a>
            </td>
            <td>
                <a href="https://medium.com/trendyol-tech/deploying-a-large-language-model-llm-with-tensorrt-llm-on-triton-inference-server-a-step-by-step-d53fccc856fa">tensorRT with TIS</a>
            </td>
            <td>Linux-only</td>
        </tr>
        <tr>
            <td>Llamafile</td>
            <td> </td>
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
            <td>DeepSparse</td>
            <td><a href="https://github.com/neuralmagic/deepsparse">Github</a></td>
            <td> </td>
            <td>CPU-only, Linux-only</td>
        </tr>
        <! ----------------- 1.3 SDKs ------------------>
        <tr><th colspan=4; style="height:100%; width:100%; text-align:center; font-size:30px"> </th></tr>
        <tr>
            <td style="font-size:20px">SDK</td>
            <td style="font-size:20px">Documentation</td>
            <td style="font-size:20px">Examples</td>
            <td style="font-size:20px">Comment</td>
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
            <th colspan=4; style="text-align:center; font-size:20px"> 
            - Further readings - 
            </th>
        </tr>
        <tr>
            <td colspan=4;>
                <a href="https://docs.llamaindex.ai/en/stable/module_guides/models/llms.html#modules">List of LLM frameworks</a>,
                <a href="https://github.com/aws-samples/amazon-sagemaker-generativeai">AWS GenAI tutorials</a>,
                <a href="https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard">Open LLM Huggingface leaderboard</a>,
                <a href="https://huggingface.co/spaces/mteb/leaderboard">MTEB leaderboard</a>,
                <a href="https://github.com/hamelsmu/llama-inference">hamelsmu llama-inference</a>,
                <a href="https://huggingface.co/spaces/mike-ravkine/can-ai-code-results">can-ai-code-results</a>,
            </td>
        </tr>
        <tr>
            <th colspan=4; style="text-align:center; font-size:30px">
             * 
            </th>
        </tr>
        <tr>
            <th colspan=4; style="text-align:center; font-size:30px">
            </th>
        </tr>
    </tbody>
    <! -------------- 2. Compression, Quantization, Pruning -------------->
    <thead>
        <tr>
            <th colspan=4; style="text-align:center; font-size:30px">
            2. Compression, Quantization, Pruning
            </th>
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
        <tr>
            <td>EETQ</td>
            <td>
                <a href="https://github.com/NetEase-FuXi/EETQ">Github</a>
            </td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>ATOM</td>
            <td></td>
            <td></td>
            <td>
                <a href="https://arxiv.org/pdf/2310.19102.pdf">2310</a>
            </td>
        </tr>
        <tr>
            <th colspan=4; style="text-align:center; font-size:30px">
            *
            </th>
        </tr>
        <tr>
            <th colspan=4; style="text-align:center; font-size:30px">
            </th>
        </tr>
    </tbody>
    <! ----------------- 3. Evaluation ------------------>
    <thead>
        <tr>
            <th colspan=4; style="text-align:center; font-size:30px">
            3. Evaluation
            </th>
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
            <td>LLM-autoeval</td>
            <td> </td>
            <td> </td>
            <td>
                <a href="https://github.com/mlabonne/llm-autoeval">Github</a>
            </td>
        </tr>
        <tr>
            <td>Deepeval</td>
            <td> </td>
            <td>
                <a href="https://docs.confident-ai.com/docs/integrations-huggingface">Integration in Huggingface Trainer</a>
            </td>
            <td>
                <a href="https://github.com/confident-ai/deepeval">Github</a>
            </td>
        </tr>
        <tr>
            <th colspan=4; style="text-align:center; font-size:30px">
            *
            </th>
        </tr>
        <tr>
            <th colspan=4; style="text-align:center; font-size:30px">
            </th>
        </tr>
    </tbody>
    <! ----------------- 4. Prompt Engineering ------------------>
    <thead>
        <tr>
            <th colspan=4; style="text-align:center; font-size:30px">
            4. Prompt Engineering
            </th>
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
        <tr>
            <th colspan=4; style="text-align:center; font-size:30px">
            *
            </th>
        </tr>
        <tr>
            <th colspan=4; style="text-align:center; font-size:30px">
            </th>
        </tr>
    </tbody>
    <! ----------------- 5. Data Ingestion ------------------>
    <thead>
        <tr>
            <th colspan=4; style="text-align:center; font-size:30px">
            5. Data Ingestion
            </th>
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
            <td>Retrieval Aware Fine-tuning (RAFT)</td>
            <td> 
                <a href="https://github.com/ShishirPatil/gorilla/tree/main/raft">
                    <div style="text-align: center;">Github</div>
                </a>
            </td>
            <td> </td>
            <td>
                <a href="https://huggingface.co/papers/2403.10131">
                    <div style="text-align: center;">2403</div>
                </a>
            </td>
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
        <tr>
            <td>Fill-In-The-Middle (FIM) transformation</td>
            <td> </td>
            <td> </td>
            <td>
                <a href="https://arxiv.org/pdf/2207.14255.pdf">
                    <div style="text-align: center;">2207</div>
                </a>
            </td>
        </tr>
        <tr>
            <th colspan=4; style="text-align:center; font-size:30px">
            *
            </th>
        </tr>
        <tr>
            <th colspan=4; style="text-align:center; font-size:30px">
            </th>
        </tr>
    </tbody>
    <! ----------------- 6. Retrieval-Augmented Generation ------------------>
    <thead>
        <tr>
            <th colspan=4; style="text-align:center; font-size:30px">
            6. Retrieval-Augmented Generation
            </th>
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
        <tr>
            <th colspan=4; style="text-align:center; font-size:30px">
            *
            </th>
        </tr>
        <tr>
            <th colspan=4; style="text-align:center; font-size:30px">
            </th>
        </tr>
    </tbody>
    <! ----------------- 7. Finetuning ------------------>
    <thead>
        <tr>
            <th colspan=4; style="text-align:center; font-size:30px">
            7. Finetuning
            </th>
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
            <td>
                <a href="https://arxiv.org/pdf/2305.18290">
                    <div style="text-align: center;">2305</div>
                </a>
            </td>
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
        <tr>
            <td>GaLore: Gradient Low Rank Projection</td>
            <td> </td>
            <td> </td>
            <td>
                <a href="https://arxiv.org/pdf/2403.03507">
                    <div style="text-align: center;">2403</div>
                </a>
            </td>
        </tr>
        <tr>
            <td>ORPO: Odds Ratio Preference Optimization</td>
            <td> </td>
            <td> </td>
            <td>
                <a href="https://arxiv.org/pdf/2403.07691.pdf">
                    <div style="text-align: center;">2403</div>
                </a>
            </td>
        </tr>
        <tr>
            <td>DNO: Direct Nash Optimization</td>
            <td> </td>
            <td> </td>
            <td>
                <a href="https://arxiv.org/pdf/2404.03715">
                    <div style="text-align: center;">2404</div>
                </a>
            </td>
        </tr>
        <! ----------------- 7.2 Frameworks ------------------>
        <tr>
            <th colspan=4; style="text-align:center; font-size:30px">
            </th>
        </tr>
        <tr>
            <td style="font-size:20px">Framework</td>
            <td style="font-size:20px">Documentation</td>
            <td style="font-size:20px">Examples</td>
            <td style="font-size:20px">Comment</td>
        </tr>
            <td>TRL</td>
            <td>
                <a href="https://github.com/huggingface/trl">
                Github
                </a>,
                <a href="https://github.com/huggingface/trl/tree/main/examples/scripts">
                Finetuning scripts
                </a>, 
            </td>
            <td> </td>
            <td> </td>
        </tr>
        <tr>
            <td>Axolotl</td>
            <td>
                <a href="https://github.com/OpenAccess-AI-Collective/axolotl">
                Github
                </a>,
                <a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/cli/train.py">
                Finetuning script
                </a>, 
            </td>
            <td> </td>
            <td>Based on TRL, Multi-GPU with Accelerate</td>
        </tr>
        <tr>
            <td>HF alignment handbook</td>
            <td>
                <a href="https://github.com/huggingface/alignment-handbook/tree/main">
                Github
                </a>,
                <a href="https://github.com/huggingface/alignment-handbook/tree/main/scripts">
                Finetuning scripts
                </a>, 
            </td>
            <td> </td>
            <td>Based on TRL, Multi-GPU with Accelerate</td>
        </tr>
        <tr>
            <td>Adapters</td>
            <td>
                <a href="https://github.com/adapter-hub/adapters/tree/main">
                Github
                </a>,
                <a href="https://github.com/adapter-hub/adapters/tree/main/examples/pytorch">
                Train adapters around LLM or BERT models
                </a>, 
            </td>
            <td> </td>
            <td>Based on TRL, Multi-GPU with Accelerate</td>
        </tr>
        <tr>
            <th colspan=4; style="text-align:center; font-size:30px">
            *
            </th>
        </tr>
        <tr>
            <th colspan=4; style="text-align:center; font-size:30px">
            </th>
        </tr>
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


## Architectures


| ![TGI-langchain-architecture](/imgs/TGI-langchain-architecture.jpeg) | 
|:--:| 
| *Architecture based on TGI and langchain proposed in this [blog post](https://www.ideas2it.com/blogs/deploying-llm-powered-applications-in-production-using-tgi)* |




## References
- [llm-course](https://github.com/mlabonne/llm-course/tree/main) Github repository
- [langchain](https://github.com/langchain-ai/langchain/tree/master) Github repository
- [llm-autoeval](https://github.com/mlabonne/llm-autoeval) Github repository
- [awesome-llms-fine-tuning](https://github.com/Curated-Awesome-Lists/awesome-llms-fine-tuning) Github repository
- [cookbook](https://github.com/huggingface/cookbook/tree/main) Github repository



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


