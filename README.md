---

<div align="center">  
  
# Do LLMs Really Struggle at NL-FOL Translation? Revealing their Strengths via a Novel Benchmarking Strategy   
[![Paper](https://img.shields.io/badge/Paper-AAAI%202026-orange)](https://ojs.aaai.org/index.php/AAAI/article/view/40258/44219)
[![arXiv](https://img.shields.io/badge/Extended%20version-arXiv-red)](https://arxiv.org/pdf/2511.11816)

 
</div>

<p align="center">
<img src="https://github.com/dslab-uniud/NL-FOL-LT/blob/main/graphical_abstract.png" alt="Graphical abstract" />
</p>


## Description   
This repository contains the code related to our paper "Do LLMs Really Struggle at NL-FOL Translation? Revealing their Strengths via a Novel Benchmarking Strategy", authored by Andrea Brunello, Luca Geatti, Michele Mignani, Angelo Montanari, and Nicola Saccomanno.


## Code to reproduce the experiments
We present here the code and the instructions relative to the dataset D_FOLIO: the other dataset used in the paper, (D_stanford), is not available since the original authors chose to keep it private; nevertheless, the pipeline we developed for its preprocessing is very similar to the one of FOLIO.

### Project Structure

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
    - **Embeddings_tasks/**
        - embeddings_Qwen_Gemini.py : Python script to run the most similar and ranking tasks using embedding-centric models (QWEN3-8B and Gemini-Embedding-001).
                                      The script runs Ranking task by default. If you want to run Most Similar, please comment/uncomment lines as described within the file.
                                      At the beginning of the file, a comment describes the two manners in which the code can be run: compute or evaluate.


- **perturbations and translations/**  
    Resources and scripts for generating the data perturbations and the translations used in the experiments. The directory contains:
    - Constant_meaning_FOLIO.pkl , Relational_meaning_FOLIO.pkl : the ontology used for each story in FOLIO 
    - **Most_similar/**
        - list_modifications_ms_FOLIO.pkl : pertubed instances considere in our experiments (\mathcal{F}_{ms} sets)
        - list_shuffled_positions_ms_FOLIO.pkl : indexes that shuffle the \mathcal{F}_{ms} sets
        - list_translations_ms_FOLIO.pkl : translations of \mathcal{F}_{ms} sets (\mathcal{T}_{ms})
    - **Ranking/**
        - list_modifications_r_FOLIO.pkl : pertubed instances considere in our experiments (\mathcal{F}_{r} sets)
        - list_shuffled_positions_r_FOLIO.pkl : indexes that shuffle the \mathcal{F}_{r} sets
        - list_translations_r_FOLIO.pkl : translations of \mathcal{F}_{r} sets (\mathcal{T}_{r})


- **utils**  
    If you are interested in the comparisons with different metrics (such as LE score or BLEU score), please download the LogicLLAMA repository (https://github.com/gblackout/LogicLLaMA.git) and put it into the folder 'utils'
    utility.py is the Python script with utility functions needed to run our code
    



### Usage

1. Navigate to the `experiments/` folder
2. Choose an experiment
3. Run the Jupyter Notebook within the experiment directory


### Notes

- The Stanford dataset is not included due to access restrictions.
- All necessary scripts and resources for reproducing the experiments with FOLIO are provided.


### Citation
If you use anything from our paper or code, please cite our work using the following format:
```
@article{brunello2025llms,
  title={Do LLMs Really Struggle at NL-FOL Translation? Revealing their Strengths via a Novel Benchmarking Strategy},
  author={Brunello, Andrea and Geatti, Luca and Mignani, Michele and Montanari, Angelo and Saccomanno, Nicola},
  journal={arXiv preprint arXiv:2511.11816},
  year={2025}
}
```
