# Question Answering in COVID-19 related scientific articles
The COVID-19 pandemic has spurred an exponential growth in scientific literature, necessitating advanced techniques for information retrieval and comprehension. In this repository, we include all scripts used for the investigation of the efficacy of Large Language Models (LLMs) for question answering (QA) tasks in the domain of COVID-19 research literature. 

## Dataset
This project makes use of three different datasets. The first one is COVID-QA, a QA dataset structured in the style of SQuAD. It comprises 2019 question-answer pairs meticulously annotated by medical experts, derived from scientific articles within the COVID-19 Open Research Dataset (CORD-19) collection. For more information the dataset is available in [COVID-QA Repository](https://github.com/deepset-ai/COVID-QA). The second one is SQuAD, a QA dataset containing over 80000 questions on Wikipedia articles, where the answer is a segment (or span) of text in the article in question. For more information and access to the dataset, refer to [The Stanford Question Answering Dataset](https://rajpurkar.github.io/SQuAD-explorer/). Finally, the last dataset was developed by us, combining the training instances of the two previously mentioned datasets. Although the development and testing remained the same across all experiments, we made the decission to keep a copy on each of the folders for the sake of simplicity on the implementation.

All data files were not uploaded to this repository due to capacity limitations.

## Scripts
                                                               
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
    ├── adapt_json.py
    └── train_dev_test.py                                                    

## Results
