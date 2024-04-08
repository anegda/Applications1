# Question Answering in COVID-19 related scientific articles
The COVID-19 pandemic has spurred an exponential growth in scientific literature, necessitating advanced techniques for information retrieval and comprehension. In this repository, we include all scripts used for the investigation of the efficacy of Large Language Models (LLMs) for question answering (QA) tasks in the domain of COVID-19 research literature. 

## Dataset
This project makes use of three different datasets. The first one is COVID-QA, a QA dataset structured in the style of SQuAD. It comprises 2019 question-answer pairs meticulously annotated by medical experts, derived from scientific articles within the COVID-19 Open Research Dataset (CORD-19) collection. For more information the dataset is available in [COVID-QA Repository](https://github.com/deepset-ai/COVID-QA). The second one is SQuAD, a QA dataset containing over 80000 questions on Wikipedia articles, where the answer is a segment (or span) of text in the article in question. For more information and access to the dataset, refer to [The Stanford Question Answering Dataset](https://rajpurkar.github.io/SQuAD-explorer/). Finally, the last dataset was developed by us, combining the training instances of the two previously mentioned datasets. Although the development and testing remained the same across all experiments, we made the decission to keep a copy on each of the folders for the sake of simplicity on the implementation.

All data files were not uploaded to this repository due to capacity limitations.

## Scripts

The structure of the repository is the following:
                                                               
    ├── data                                  # Folder containing all data used
    |   ├── Combined
    |   │   ├── dev.json                      # COVID-QA dev data 
    |   │   ├── test.json                     # COVID-QA test data
    |   │   └── train.json                    # Union of the COVID-QA and SQuAD train splits
    |   ├── COVID-QA
    |   │   ├── COVID-QA.json                 # Original COVID-QA data
    |   │   ├── data.json                     # COVID-QA data, processed to have the same format as SQuAD
    |   │   ├── dev.json                      # COVID-QA dev data
    |   │   ├── test.json                     # COVID-QA test data
    |   │   └── train.json                    # COVID-QA train data
    |   ├── SQuAD
    |   │   ├── dev.json                      # COVID-QA dev data
    |   │   ├── test.json                     # COVID-QA test data 
    |   │   ├── train.json                    # SQuAD train data
    |   │   └── validation.json               # Original SQuAD validation data (unused)
    |   └── wandb
    ├── adapt_json.py                         # Script used to unify the format between the COVID-QA and SQuAD data
    ├── combine.py                            # Script used to merge the training data from COVID-QA and SQuAD together
    ├── qa_finetuning.py                      # Original script used to fine-tune and evaluate the models
    ├── qa_finetuning_hyp_search.py           # Script that performs a search of the best hyperparameters for a model
    ├── train.slurm                           # Slurm launcher for the fine-tuning task
    └── train_dev_test.py                     # Given a dataset, split it into a train, dev and test splits                               

## Results

To evaluate which of the training dataset would yield the best results, a [FacebookAI/roberta-base](https://huggingface.co/) model was fine-tuned using each dataset and evaluated on the COVID-QA dev and test data. The results obtained reveal that the **XXXXX** dataset yielded the best results, with a **XXXXX** exact match score and a **XXXXX** f1-score.

Then, three different models where fine-tuned with the best performing training data. These models were: 
* [roberta-base](https://huggingface.co/FacebookAI/roberta-base): not trained on Question Answering, baseline model. Same model that was used on the previous step.
* [roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2): model specifically trained for Question Answering on the [SQuAD 2.0 dataset](https://huggingface.co/datasets/rajpurkar/squad_v2).
* [roberta-base-squad2-nq-bioasq](https://huggingface.co/scite/roberta-base-squad2-nq-bioasq): a version of roberta-base-squad2 trained on Question Answering and the BioASQ 10B medical dataset.
The best performing one ended up being **XXXXX**, with a **XXXXX** exact match score and a **XXXXX** f1-score.

For the last step, we did a hyperparemeter optimization on this model. After finding the best combination of parameters, its performance increased by **XXXXX**%, achieving a **XXXXX** exact match score and a **XXXXX** f1-score.
