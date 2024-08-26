import os
import json
from random import random
def main() -> None:
    entrys = []
    for file in os.listdir("./dataset"):
        file_name = os.fsdecode(file)
        entry = {}
        entry["file"] = file_name
        entry["label"] = 1 if file_name.startswith("rosettalia") else 0
        entry["type"] = "train" if random() < 0.8 else "val"
        entrys.append(entry)
    datafile = {}
    datafile["data"] = entrys
    print(f"{len(entrys)} files found")

    output = open("config.json",'+wt')
    json.dump(datafile, output)
    output.close()

if __name__ == "__main__":
    main()