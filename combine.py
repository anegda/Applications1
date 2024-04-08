import json
import random

file1 = "/tartalo03/users/udixa/qa_applications1/data/COVID-QA/train.json"
file2 = "/tartalo03/users/udixa/qa_applications1/data/SQuAD/train.json"
new_file = "/tartalo03/users/udixa/qa_applications1/data/Combined/train.json"

with open(file1, "r") as f1:
    data1 = json.load(f1)

with open(file2, "r") as f2:
    data2 = json.load(f2)

print("COVID:", len(data1["data"]))
print("SQuAD:", len(data2["data"]))
print("RIGHT SUM:", str(len(data1["data"])+len(data2["data"])), "\n")

combined_data = data1["data"] + data2["data"]
random.shuffle(combined_data)
new_json = {"data": combined_data}
print("NEW", new_json["data"][0].keys(), len(new_json["data"]))

with open(new_file, "w") as f3:
    json.dump(new_json, f3, indent=4)