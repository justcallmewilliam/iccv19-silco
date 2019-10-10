json_file = "cl_voc07.json"
json_file = "cl_coco14.json"
import json
simple_json_file = "simple.json"

ss = dict()
with open(json_file, 'r') as f:
    data_json = json.load(f)
    for key,item in data_json.items():

    print("ok")
    print("ok")