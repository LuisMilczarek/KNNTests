import os
import json
from random import randint
def main() -> None:
    # entrys = []
    PROPORTION_TRAIN_VAL = 0.8
    datafile = {}
    datafile["labels"] = ["amarantus","lefantopus","lenguavaca","rosettalia","trevolus"]
    datafile["summary"] = {"train":{},"val":{}}
    datafile["data"] = {"train":{},"val":{}}
    files = {}
    files_val_index = {}

    for label in datafile["labels"]:
        files[label] = []
        files_val_index[label] = []
        for type in datafile["summary"].keys():
            datafile["summary"][type][label] = 0
        for type in datafile["data"].keys():
            datafile["data"][type][label] = []
    for file in os.listdir("./dataset"):
        file_name = os.fsdecode(file)
        # entry = {}
        # entry["file"] = file_name
        for i, label in enumerate(datafile["labels"]):
            if file_name.startswith(label):
                files[label].append(file_name)
                break
    
    for label in datafile["labels"]:
        size = len(files[label])
        val_size = int( size * (1- PROPORTION_TRAIN_VAL))
        counter = 0
        while counter <= val_size:
            index = randint(0,size-1)
            if not index in files_val_index[label]:
                files_val_index[label].append(index)
                counter += 1
    
    print(datafile)
    
    for label in files.keys():
        for i, file in enumerate(files[label]):
            type = "val" if i in files_val_index[label] else "train"
            datafile["data"][type][label].append(file)
            datafile["summary"][type][label] += 1



    print(datafile)

    output = open("config.json",'+wt')
    json.dump(datafile, output)
    output.close()

if __name__ == "__main__":
    main()