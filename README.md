<div id="top"></div>

<br />
<div align="center">
  <a href="https://github.com/Peltarion/thesis-eliott">
    <img src="images/peltarion_logotype_Pi_red.svg" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">MSc Thesis Project by Eliott Remmer</h3>

  <p align="center">
    Explainability Methods for Transformer-based Networks: a Comparative Analysis
    <br />
    <a href="https://github.com/Peltarion/thesis-eliott"><strong>Explore the docs »</strong></a>
    <br />
    <a href="https://github.com/Peltarion/thesis-eliott/issues">Report Bug</a>
  </p>
</div>



<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#abstract">Abstract</a></li>
        <li><a href="#keywords">Keywords</a></li>
      </ul>
    </li>
    <li>
      <a href="#about-the-repository">About The Repository</a>
      <ul>
        <li><a href="#content">Content</a></li>
        <li><a href="#environment-variables">Environment variables</a></li>
      </ul>
    </li>
    <li>
      <a href="#running-experiments">Running Experiments</a>
      <ul>
        <li><a href="#data-exploration">Data Exploration</a></li>
        <li><a href="#preprocessing">Preprocessing</a></li>
        <li><a href="#fine-tuning">Fine-tuning</a></li>
        <li><a href="#benchmarks">Benchmarks</a></li>
        <li><a href="#explanation-evaluation">Explanation Evaluation</a></li>
      </ul>
    </li>
     <li>
      <a href="#references">References</a>
    </li>
  </ol>
</details>


## About The Project

<p align="right">(<a href="#top">back to top</a>)</p>

### Abstract

Alongside the increase in complexity of Artificial Intelligence (AI) models follows an increased diﬀiculty to interpret predictions made by the models. This thesis work provides insights and understanding of the differences and similarities between explainability methods for AI models. Cracking open black-box models is important, especially if AI is applied in sensitive domains such as to e.g. aid medical professionals. In recent years, the use of Transformer based architectures such as Bidirectional Encoder Representations from Transformers (BERT) has become common in the field of Natural Language Processing (NLP), showing human-level performance on tasks such as sentiment classification and question answering. A growing portion of research within eXplainable AI (XAI) has shown success in using explainability methods to output auxiliary explanations at inference time together with predictions made by these complex models. There is a distinction to be made whether the explanations emerge as a part of the prediction or subsequently via a separate model. These two categories of explainability methods are referred to as self-explaining and post-hoc. The goal of this work was to evaluate, analyze and compare these two categories of methods for assisting BERT models with explanations in the context of sentiment classification. A comparative analysis was conducted in order to investigate quantitative and qualitative differences. To measure the quality of explanations, the Intersection Over Union (IOU) and Precision-Recall Area Under the Curve (PR-AUC) scores were used together with Explainable NLP (ExNLP) datasets containing human annotated explanations. Apart from discussing benefits, drawbacks and assumptions of the different methods, results showed that the self-explaining method proved successful on longer input texts while the post-hoc method performed better on shorter input texts. Given the subjective nature of explanation quality, the work should be extended in several directions proposed in this work in order to fully capture the nuances of the explainability methods.

### Keywords

XAI, NLP, Transformers, BERT, explainable predictions, SHAP, Attention


## About The Repository

<p align="right">(<a href="#top">back to top</a>)</p>

### Content

- `/data`

  Contains the two ExNLP datasets used in this thesis. `/movies` contains the Movie Reviews dataset [[1]](#1) and `/twitter-sentiment-extraction` contains the Twitter Sentiment Extraction dataset [[2]](#2).

- `/notebooks`

  Folder contains all notebooks used in the project. `data_exploration.ipynb` contains some basic data exploration. `compare_explanations.ipynb` was used for visualizing and qualitatively observing explanations. `wandb_sweeps.ipynb` was used for a baysian optimized hyperparameter sweep with Weights and Biases [[3]](#3).

- `/scripts`

  Contains a scipt to run pre-processing of the Twitter Sentiment Extraction data `preprocess_twitter.py` as well as a script to sort the environment file.

- `/thesis_eliott`

  Main folder containing scripts to run fine-tuning `train.py`, benchmarking classifiers `benchmark.py` `baselines.py` and explanation evaluation `explain.py`. Raw results from benchmarking and explanation evaluation are found in `/results_baseline` and `/results_explain`.  Useful functions for explanation extraction and evaluation are contained in `utilities.py`.
  
- `environment.yml`

  File contains all the necessary dependencies for running.


### Environment variables
- `DATA_DIR`

  The absolute path to where you want to store data on the host machine. Commonly set to the data folder in this repository, such as `/mnt/storage/data/<first.lastname>/<project-name>/data` when working on `gpugpu` or `scorpion`.


- `MODEL_DIR`

  The absolute path to where you want to store models on the host machine. Commonly set to the data folder in this repository, such as `/mnt/storage/data/<first.lastname>/<project-name>/model` when working on `gpugpu` or `scorpion`.


- `GPU_IDS`

  The GPU ids that will be exposed inside the container as indexed by running `nvidia-smi` on the host machine. Observe that inside the container the indices will be start at 0. If needed, multiple GPUs can be used as: `GPU_IDS=0,1`


- `JUPYTER_PW`

  The password to the Jupyter lab service. You choose this yourself.


- `JUPYTER_PORT`

  The port to the Jupyter lab service. You choose this yourself, but it is a good idea to runt `docker ps`  and check which ports are already taken before you settle on one.


- `TENSORBOARD_PORT`

  The port to the Tensorboard service. You choose this yourself, but consider the same issues as when setting `JUPYTER_PORT`.


- `NEPTUNE_PROJECT`

  Used by Neptune experiment logger to indicate where your experiment metrics will be stored.
  To see how to setup a Neptune project and API token, see
  https://docs.neptune.ai/getting-started/installation#authentication-neptune-api-token


- `NEPTUNE_API_TOKEN`

  API token for using Neptune experiment logger.

## Running Experiments

<p align="right">(<a href="#top">back to top</a>)</p>

### Data Exploration

Run different cells in `data_exploration.ipynb` to extract information about input length.

### Preprocessing

Run `preprocess_twitter.py` to execute some preprocessing on the tweets data.

### Fine-tuning

For hyperparameter tuning, run `wandb.ipynb` to initiate a Weights and Biases sweep with the parameter ranges specified in the notebook. For model fine-tuning, run `train.py`, specifying parameters, dataset and pre-tranied model in the Python file. Do also specify metrics and Neptune API token to enable logging with neptune.

### Benchmarks

To reproduce results from the different benchmarks, run `benchmark.py`. The n-grams on character-level and work-level for the TF-IDF+Linear classifiers are specified in the `bow_linear` function in `baselines.py` as well as the strategy for the `dummy` classifier.

### Explanation Evaluation

In `explain.py`, start by specifying the parameters of the evaluation in the top of the `if __name__ == "__main__":` clause. This includes defining e.g. `DATASET`, `METHOD` and `MODEL`. Run the file to initate the evaluation, saving results in `PATH_TO_RESULTS`.

## References

<a id="1">[1]</a> 
S. Lundberg and S. I. Lee, (2017),
“A Unified Approach to Interpreting Model Predictions”,
Advances in Neural Information Processing Systems, vol. 2017-December, pp. 4766–4775,
https://arxiv.org/abs/1705.07874v2

<a id="2">[2]</a>
Kaggle, (2020),
“Tweet Sentiment Extraction: Extract support phrases for sentiment labels”,
https://www.kaggle.com/competitions/tweet-sentiment-extraction/overview

<a id="3">[3]</a>
L. Biewald, (2020),
“Experiment Tracking with Weights and Biases”,
https://www.wandb.com/


<p align="right">(<a href="#top">back to top</a>)</p>
