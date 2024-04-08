# Question Answering in COVID-19 related scientific articles
The COVID-19 pandemic has spurred an exponential growth in scientific literature, necessitating advanced techniques for information retrieval and comprehension. In this repository, we include all scripts used for the investigation of the efficacy of Large Language Models (LLMs) for question answering (QA) tasks in the domain of COVID-19 research literature. 

# Dataset
The dataset utilized in this project is COVID-QA, a QA dataset structured in the style of SQuAD. It comprises 2019 question-answer pairs meticulously annotated by medical experts, derived from scientific articles within the COVID-19 Open Research Dataset (CORD-19) collection. For more information the dataset is available in [COVID-QA Repository]{https://github.com/deepset-ai/COVID-QA}.

# Scripts
                                                               
    ├── data                                                           
    │   └── COVID-QA.json                                          
    │   ├── data.json                                       
    │   ├── dev.json   
    │   ├── test.json   
    │   └── train.json                                          
    ├── adapt_json.py
    └── train_dev_test.py                                                    

# Results
