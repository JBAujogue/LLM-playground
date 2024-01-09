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
            <td>Huggingface `transformers`</td>
        </tr>
        <tr>
            <td>Optimum</td>
        </tr>
    </tbody>
    <! ----------------- 2. Quantization ------------------>
    <thead>
        <tr>
            <th colspan=4; style="height:100%; width:100%; text-align:center; font-size:30px">2. Quantization</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="font-size:20px">Method</td>
            <td style="font-size:20px">Documentation</td>
            <td style="font-size:20px">Examples</td>
            <td style="font-size:20px">Paper</td>
        </tr>
        <! AWQ> 
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
        <! GPTQ> 
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
        <! BitsAndBytes> 
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
    </tbody>
    <! ----------------- 3. Evaluation ------------------>
    <thead>
        <tr>
            <th colspan=4; style="height:100%; width:100%; text-align:center; font-size:30px">3. Evaluation</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="font-size:20px">Benchmark</td>
            <td style="font-size:20px">Tasks</td>
            <td style="font-size:20px">Data</td>
            <td style="font-size:20px">Leaderboard</td>
        </tr>
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
    </tbody>
    <! ----------------- 5. Retrieval-Augmented Generation ------------------>
    <thead>
        <tr>
            <th colspan=4; style="height:100%; width:100%; text-align:center; font-size:30px">5. Retrieval-Augmented Generation</th>
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
            <td>self-RAG</td>
        </tr>
    </tbody>
    <! ----------------- 6. Dataset construction ------------------>
    <thead>
        <tr>
            <th colspan=4; style="height:100%; width:100%; text-align:center; font-size:30px">6. Dataset construction</th>
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
            <td>LLAMA-pro: Progressive Learning of LLMs</td>
            <td> </td>
            <td> </td>
            <td> </td>
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
            <td>DPO: Direct Preference Optimization</td>
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
            <td>LoRA: Low Ranking Adaptation</td>
            <td> </td>
            <td> </td>
            <td> </td>
        </tr>
        <tr>
            <td>PEFT: Parameter-Efficient FineTuning</td>
            <td> </td>
            <td> </td>
            <td> </td>
        </tr>
    </tbody>
</table>





## General references
- [llm-course](https://github.com/mlabonne/llm-course/tree/main) Github repository
- [langchain](https://github.com/langchain-ai/langchain/tree/master) Github repository


