# Do LLMs Really Struggle at NL-FOL Translation?  
_Revealing their Strengths via a Novel Benchmarking Strategy_

This repository contains the code for the article **"Do LLMs Really Struggle at NL-FOL Translation? Revealing Strengths and Weaknesses via a Novel Benchmarking Strategy"**.
We present here the code relative to the dataset D_FOLIO: the other dataset used in the paper, (D_stanford), is not available since the original authors chose to keep it private; nevertheless, the pipeline we developed for its preprocessing is very similar to the one of FOLIO.

## Project Structure

- **datasets/**  
    - **FOLIO**: Public dataset available to the link: https://huggingface.co/datasets/yale-nlp/FOLIO
    - **Stanford**: Not publicly available


- **experiments/**  
    Contains code and resources for the experiments:
    - **Logical Translation/**
    - **Most Similar/**
    - **Ranking/**  
    Each of these three subfolders includes:
        - Notebook with the code to run the experiment: the code to call the OpenAI's API is commented. Follow the directions provided in the code to uncomment the code and call the API.
        - `Request/`: Input prompts and data we issued through the OpenAI's API (2 files: GPT-4o-mini and o3-mini)
        - `Result/`: Outputs we received through the OpenAI's API (2 files: GPT-4o-mini and o3-mini)
            # The subdirectory "Clean Results" contains the postprocessed outputs used within our experiments
    - **Embeddings_tasks/**
        # embeddings_Qwen_Gemini.py : Python script to run the most similar and ranking tasks using embedding-centric models (QWEN3-8B and Gemini-Embedding-001).
                                      The script runs Ranking task by default. If you want to run Most Similar, please comment/uncomment lines as described within the file.
                                      At the beginning of the file, a comment describes the two manners in which the code can be run: compute or evaluate.

        # **Embedding : utility files to be used for embedding calculations


- **perturbations and translations/**  
    Resources and scripts for generating the data perturbations and the translations used in the experiments. The directory contains:
    - Constant_meaning_FOLIO.pkl , Relational_meaning_FOLIO.pkl : the ontology used for each story in FOLIO 
    - **Most_similar/**
        # list_modifications_ms_FOLIO.pkl : pertubed instances considere in our experiments (\mathcal{F}_{ms} sets)
        # list_shuffled_positions_ms_FOLIO.pkl : indexes that shuffle the \mathcal{F}_{ms} sets
        # list_translations_ms_FOLIO.pkl : translations of \mathcal{F}_{ms} sets (\mathcal{T}_{ms})
    - **Ranking/**
        # list_modifications_r_FOLIO.pkl : pertubed instances considere in our experiments (\mathcal{F}_{r} sets)
        # list_shuffled_positions_r_FOLIO.pkl : indexes that shuffle the \mathcal{F}_{r} sets
        # list_translations_r_FOLIO.pkl : translations of \mathcal{F}_{r} sets (\mathcal{T}_{r})


- **utils**  
    If you are interested in the comparisons with different metrics (such as LE score or BLEU score), please download the LogicLLAMA repository (https://github.com/gblackout/LogicLLaMA.git) and put it into the folder 'utils'
    utility.py is the Python script with utility functions needed to run our code
    



## Usage

1. Navigate to the `experiments/` folder
2. Choose an experiment
3. Run the Jupyter Notebook within the experiment directory


## Notes

- The Stanford dataset is not included due to access restrictions.
- All necessary scripts and resources for reproducing the experiments with FOLIO are provided.

