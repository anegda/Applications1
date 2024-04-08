import argparse
import json
from collections import defaultdict, OrderedDict
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, TrainingArguments, set_seed
from transformers import AutoTokenizer
from transformers import default_data_collator

from trainer_qa import QuestionAnsweringTrainer

SPLITS = ['train', 'dev', 'test']
SEED = 123
squad_v2 = False  # IF USING THE SECOND VERSION OF THE DATASET, WHERE THE IMPOSSIBLE ANSWER IS POSSIBLE


def postprocess_qa_predictions(
        examples, features,
        raw_predictions,
        tokenizer,
        n_best_size=20, max_answer_length=30,
):
    all_start_logits, all_end_logits = raw_predictions
    # MAP EACH FEATURE TO THEIR EXAMPLE USING example_id
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # DICTIONARIES TO FILL
    predictions = OrderedDict()

    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # ITERATE OVER ALL THE EXAMPLES
    for example_index, example in enumerate(tqdm(examples)):
        # FEATURES ASSOCIATED TO THE CURRENT EXAMPLE
        feature_indices = features_per_example[example_index]

        min_null_score = None  # Only used if squad_v2 is True.
        valid_answers = []

        context = example["context"]
        # ITERATE OVER FEATURES
        for feature_index in feature_indices:
            # MODEL PREDICTIONS FOR THIS FEATURE
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # MAP OF THE LOGITS TO THE ORIGINAL TEXT
            offset_mapping = features[feature_index]["offset_mapping"]

            # IN CASE ONE FEATURE PREDICTS THE CLS TOKEN AS THE ANSWER (BECAUSE THE ANSWER IS NOT IN ITS CONTEXT),
            # SET IT AS THE MIN SCORE
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # HOW TO CHOOSE 2ND BEST ANSWER? 2ND BEST BEGINNING AND END?
            # BEST BEGINNING AND 2ND BEST END?
            # TO SOLVE: CALCULATE COMBINED SCORE OF ALL COMBINATIONS OF BEGINNING AND END
            start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # SKIP ANSWERS WHOSE INDEXES POINT TO TOKENS OUTSIDE THE CONTEXT OR OUT-OF-BOUNDS
                    if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or len(offset_mapping[start_index]) < 2
                            or offset_mapping[end_index] is None
                            or len(offset_mapping[end_index]) < 2
                    ):
                        continue
                    # SKIP ANSWERS WITH HIGHER THAN MAX LENGTH, OR WITH LOWER LENGTH THAN ZERO
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            # COMBINED SCORE
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char],
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )

        if valid_answers:
            # SORT ANSWERS FROM HIGHEST SCORING TO LOWEST
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # IN CASE NO ANSWER WAS NON-NULL, CREATE A FAKE ONE WITH SCORE ZERO
            best_answer = {"text": "", "score": 0.0}

        # PICK BEST NON-NULL ANSWER
        # IF squad_v2 = True, PICK IMPOSSIBLE ANSWER (CLS) IF ALL FEATURES GIVE IT A HIGH SCORE
        if not squad_v2:
            predictions[example["id"]] = best_answer["text"]
        else:
            answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
            predictions[example["id"]] = answer

    # ADAPT THE FORMAT OF THE JSON TO FIT THE REQUIREMENTS FOR THE METRIC USED (SQUAD)
    if squad_v2:
        formatted_predictions = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()]
    else:
        # PREDICTIONS FORMAT: [ {'id': int, 'prediction_text': String}, ... ]
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

    return predictions, formatted_predictions


def main(
        dataset_name: str | None,
        data_folder: Path,
        output_folder: Path,
        model_path: str,
        run_count: int,
        seed: int = SEED,
        do_hyperparameter_search: bool = False
):
    # -------------------------------------------
    # 0.- PREPARING THE ENVIRONMENT
    # -------------------------------------------
    set_seed(seed)

    dataset_name = dataset_name or Path(data_folder).name
    output_folder = output_folder / dataset_name / str(run_count)
    output_folder.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------
    # 1.- LOADING DATASET
    # -------------------------------------------

    data_files = {split: str(data_folder / f'{split}.json') for split in SPLITS}
    dataset = load_dataset('json', data_files=data_files, field='data')
    print(dataset)

    # -------------------------------------------
    # 2.- TOKENIZE THE TEXTS
    # -------------------------------------------

    # 2.1.- Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 2.2.- Prepare text and handle long contexts with a sliding window approach
    def prepare_features(examples, inference=False):
        try:
            max_length = tokenizer.model_max_length
        except:
            max_length = 512

        doc_stride = 128

        # Tokenize just the question (to get the length)
        # <s>question</s>context</s>
        # If the context is too long, it will be split into multiple samples (fragments). Ej:
        # <s>question</s>context1</s>
        # <s>question</s>context2</s>
        # The question will be the same for all the samples and won't be truncated
        tokenized_examples = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # -------------------------------------------------------------------------------------------
        # Get the start and end positions of the answer and prepare the features for training or evaluation
        # -------------------------------------------------------------------------------------------
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        if not inference:
            # FOR EVALUATION FEATURES
            # TOKEN TO CHARACTER POSITION MAP (IN THE ORIGINAL EXAMPLE)
            offset_mapping = tokenized_examples.pop("offset_mapping")

            tokenized_examples["start_positions"] = []
            tokenized_examples["end_positions"] = []

            for i, offsets in enumerate(offset_mapping):
                # LABEL IMPOSIBLE ANSWERS WITH THE INDEX OF THE CLS TOKEN
                input_ids = tokenized_examples["input_ids"][i]
                cls_index = input_ids.index(tokenizer.cls_token_id)

                # GRAB THE SEQUENCE THAT DETERMINES WHICH TOKEN IS FROM THE CONTEXT AND WHICH FROM THE QUESTION
                sequence_ids = tokenized_examples.sequence_ids(i)

                sample_index = sample_mapping[i]
                answers = examples["answer"][sample_index]

                '''
                # If no answers are given, set the cls_index as answer.
                if len(answers["answer_start"]) == 0:
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                '''

                # STARTING AND ENDING INDEX OF THE ANSWER
                start_char = answers["answer_start"]
                end_char = start_char + len(answers["text"])

                # MOVE START INDEX INTO A POSITION IN THIS SPAN OF TEXT
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                # MOVE END INDEX INTO A POSITION IN THIS SPAN OF TEXT
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # ANSWER GIVEN IS OUT OF BOUNDS, SET POSITIONS AS CLS INDEX
                if not (
                        offsets[token_start_index][0] <= start_char
                        and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # MOVE START TOKEN INDEX TO THE BEGINNING OF THE ANSWER
                    while (
                            token_start_index < len(offsets)
                            and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    # MOVE END TOKEN INDEX TO THE END OF THE ANSWER
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

            return tokenized_examples

        else:
            # FOR EVALUATION FEATURES

            tokenized_examples["example_id"] = []
            # ITERATE OVER EXAMPLES
            for i in range(len(tokenized_examples["input_ids"])):
                # STORE SEQUENCE CORRESPONDING TO THE EXAMPLE
                sequence_ids = tokenized_examples.sequence_ids(i)
                context_index = 1  # if pad_on_right else 0

                sample_index = sample_mapping[i]
                tokenized_examples["example_id"].append(examples["id"][sample_index])

                tokenized_examples["offset_mapping"][i] = [
                    (o if sequence_ids[k] == context_index else None)
                    for k, o in enumerate(tokenized_examples["offset_mapping"][i])
                ]

            return tokenized_examples

    # PREPARE THE TRAINING DATASET FEATURES FOR THE TRAINER
    training_features = dataset['train'].map(prepare_features, batched=True, remove_columns=dataset["train"].column_names)

    # PREPARE THE EVALUATION DATASET FEATURES FOR THE TRAINER
    eval_features = dataset["dev"].map(lambda _batch: prepare_features(_batch, inference=True), batched=True, remove_columns=dataset["dev"].column_names)

    # -------------------------------------------
    # 3.- FINE-TUNING THE MODEL
    # -------------------------------------------

    # POST-PROCESS THE RAW-PREDICTIONS OBTAINED BY THE MODEL
    def trainer_post_process(examples, features, model_predictions):
        """
        The trainer will use this function to post-process the raw predictions
        It expect the next parameters
        - eval_examples: Subset of the examples used for evaluation (the og dataset)
        - eval_dataset/features: The evaluation dataset
        - output.predictions
        """
        _, formatted_preds = postprocess_qa_predictions(
            examples=examples, features=features,
            raw_predictions=model_predictions, tokenizer=tokenizer
        )
        refs = [{"id": str(ex["id"]), "answers": ex['answer']} for ex in examples]
        return formatted_preds, refs

    # -------------------------------------------

    # PREPARE THE EVALUATION METRIC
    metric = evaluate.load("squad")

    def compute_metrics(p: tuple):
        preds, refs = p
        return metric.compute(predictions=preds, references=refs)

    # -------------------------------------------

    # LOAD THE MODEL

    model = AutoModelForQuestionAnswering.from_pretrained(model_path)

    args = TrainingArguments(
        str(output_folder),
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        lr_scheduler_type="linear",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    data_collator = default_data_collator
    trainer = QuestionAnsweringTrainer(
        model,
        args,
        train_dataset=training_features,
        eval_dataset=eval_features,
        data_collator=data_collator,
        tokenizer=tokenizer,
        eval_examples=dataset["dev"],
        post_process_function=trainer_post_process,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # -------------------------------------------
    # 4.- EVALUATION
    # -------------------------------------------

    # PREPARE THE VALIDATION DATA TO EXTRACT THE ANSWER FROM THE TEXT
    test_features = dataset["test"].map(lambda batch: prepare_features(batch, inference=True), batched=True, remove_columns=dataset["test"].column_names)

    # PREDICTIONS FOR ALL FEATURES
    raw_predictions = trainer.predict(predict_examples=dataset["test"], predict_dataset=test_features)

    # BY DEFAULT, Trainer HIDES UNUSED COLUMNS, SO WE SHOW THEM AGAIN
    print(test_features.set_format(type=test_features.format["type"], columns=list(test_features.features.keys())))
    test_features.set_format(type=test_features.format["type"], columns=list(test_features.features.keys()))

    ''' FUNCION FINAL '''

    # POST-PROCESS THE RAW-PREDICTIONS OBTAINED BY THE MODEL
    final_predictions, formatted_predictions = postprocess_qa_predictions(
        examples=dataset["test"], features=test_features,
        raw_predictions=raw_predictions.predictions, tokenizer=tokenizer
    )

    print(formatted_predictions)
    references_aux = [{"id": ex["id"], "answers": ex["answer"]} for ex in dataset["test"]]

    # REFERENCES FORMAT: [ {'id': int, 'answers': {'answer_start': int, 'text': String} }, ... ]
    references = []
    for ref in references_aux:
        answer_start = [ref['answers']['answer_start']]
        text = [ref['answers']['text']]

        new_ref = {
            'id': ref['id'],
            'answers': {'answer_start': answer_start, 'text': text}
        }

        references.append(new_ref)

    print(references)
    metric.compute(predictions=formatted_predictions, references=references)

    with open(output_folder / "test_results.json", "w", encoding='utf8') as f:
        json.dump(metric.compute(predictions=formatted_predictions, references=references), f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, required=False, default=None)
    parser.add_argument('--data_folder', type=Path, required=False, default=Path('/tartalo03/users/udixa/qa_applications1/data/COVID-QA'))
    parser.add_argument('--output_folder', type=Path, required=False, default=Path('/gscratch5/users/prafai/qa_applications1'))
    parser.add_argument('--model_path', type=str, required=False, default='distilbert-base-uncased')
    parser.add_argument('--seed', type=int, required=False, default=SEED)
    parser.add_argument('--run_count', type=int, required=False, default=1)
    parser.add_argument('--do_hyperparameter_search', action='store_true', default=False)
    args = parser.parse_args()

    main(**vars(args))
