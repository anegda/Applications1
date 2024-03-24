import json
import random

def split_data(original_file, train_file, dev_file, test_file, train_ratio=0.8, dev_ratio=0.1, shuffle=True):
    # Load the original JSON file
    with open(original_file, 'r') as f:
        data = json.load(f)

    # Shuffle the data if needed
    if shuffle:
        random.shuffle(data['data'])

    # Calculate the sizes of train, dev, and test sets
    total_size = len(data['data'])
    train_size = int(train_ratio * total_size)
    dev_size = int(dev_ratio * total_size)
    test_size = total_size - train_size - dev_size

    # Split the data
    train_data = data['data'][:train_size]
    dev_data = data['data'][train_size:train_size + dev_size]
    test_data = data['data'][train_size + dev_size:]

    # Save the split data as JSON files
    with open(train_file, 'w') as f:
        json.dump({'data': train_data}, f, indent=4)
    with open(dev_file, 'w') as f:
        json.dump({'data': dev_data}, f, indent=4)
    with open(test_file, 'w') as f:
        json.dump({'data': test_data}, f, indent=4)


# Example usage
split_data('/tartalo03/users/udixa/qa_applications1/data/data.json', '/tartalo03/users/udixa/qa_applications1/data/train.json', '/tartalo03/users/udixa/qa_applications1/data/dev.json', '/tartalo03/users/udixa/qa_applications1/data/test.json')
