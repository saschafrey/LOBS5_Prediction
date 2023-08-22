# Generative AI for End-to-End Limit Order Book Modelling
## A Token-Level Autoregressive Generative Model of Message Flow Using a Deep State Space Network

This repository provides the implementation for the
paper: Generative AI for End-to-End Limit Order Book Modelling. The preprint is available [here](https://arxiv.org/abs/xxxx.xxxxx).

The repository is a fork of [the original S5 repository](https://github.com/lindermanlab/S5).

Developing a generative model of realistic order flow in financial markets is a challenging open problem, with numerous applications for market participants. Addressing this, we propose the first end-to-end autoregressive generative model that generates tokenized limit order book (LOB) messages. These messages are interpreted by a Jax-LOB simulator, which updates the LOB state. To handle long sequences efficiently, the model employs \emph{simplified structured state-space layers} to process sequences of order book states and tokenized messages. Using LOBSTER data of NASDAQ equity LOBs, we develop a custom tokenizer for message data, converting groups of successive digits to tokens, similar to tokenization in large language models.

## Requirements & Installation

To install required packages, run `pip install -r requirements.txt`.

The GPU installation of JAX can cause problems, further instructions are available [here](https://github.com/google/jax#installation).

## Data Download

The data used is NASDAQ LOB data from [LOBSTER](https://lobsterdata.com/index.php).
After downloading and unpacking the data files, they need to be pre-processed for model training. An example command for GOOG is as follows:

`python lob/preproc.py --data_dir /path/to/LOBS5/data/GOOG/ --save_dir /path/to/LOBS5/data/GOOG/ --n_tick_range 500 --use_raw_book_repr`

## Repository Structure

Directories and files that ship with GitHub repo:
```
lob/                    Source code for LOB models, datasets, etc.
    ar_pred.ipynb           Test set evaluations and plotting for trained model.
    dataloading.py          Dataloading functions.
    encoding.py             Message tokenization: encoding and decoding
    evaluation.py           Model evaluation logic
    inference.py            Logic for model inference loop
    init_train.py           Train state initialisation
    lob_seq_model.py        Defines LOB deep sequence model that consist of stacks of S5
    lobster_dataloader.py   Defines dataset and dataloading
    preproc.py              Pre-processes LOBSTER data for the model
    run_eval.py             Script to run model inference with trained model
    sweep.py                Hyperparamter sweep (WanDB)
    train_helpers.py        Functions for optimization, training and evaluation steps.
    train.py                Training loop code.
    validation_helpers.py   Helper functions for model validation
s5/                     Original S5 code
bin/                    Shell scripts for downloading data and running experiments.
requirements.txt            Package requirements
run_train.py            Training loop entrypoint.
```

## Experiments / Paper Results

Paper results and plots are calculated in `lob\ar_pred.ipynb`.

## Citation
Please use the following when citing our work:
```
@article{nagy2023generative,
  doi = {},
  url = {https://arxiv.org/abs/xxxx.xxxxx},
  author = {Nagy, Peer and Frey, Sascha and Sapora, Silvia and Li, Kang and Calinescu, Anisoara and Zohren, Stefan and Foerster, Jakob},
  keywords = {generative AI, structured state space models, limit order books, ML},
  title = {Generative AI for End-to-End Limit Order Book Modelling: A Token-Level Autoregressive Generative Model of Message Flow Using a Deep State Space Network},
  publisher = {arXiv},
  year = {2023},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

Please reach out if you have any questions.
