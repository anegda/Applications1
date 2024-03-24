import json

import json


def transform_json(original_file, transformed_file):
    with open(original_file, 'r') as f:
        original_data = json.load(f)

    transformed_data = {'data': []}

    for item in original_data['data']:
        for p in item['paragraphs']:
            for qa in p['qas']:
                for ans in qa['answers']:
                    answer = ans
                    question = qa['question']
                    document = p['context']
                    id = qa['id']

                    # Create a new element in the transformed data structure
                    transformed_item = {'id': id, 'context': document, 'question': question, 'answer': answer}

                    # Append to the 'data' list
                    transformed_data['data'].append(transformed_item)
                    break

    # Save the transformed data as JSON
    with open(transformed_file, 'w') as f:
        json.dump(transformed_data, f, indent=4)


# Example usage
transform_json('/data/COVID-QA.json',
               'data/data.json')
