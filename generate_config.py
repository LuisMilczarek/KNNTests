import os
import json
def main() -> None:
    entrys = []
    for file in os.listdir("./dataset"):
        file_name = os.fsdecode(file)
        entry = {}
        entry["file"] = file_name
        entry["label"] = 1 if file_name.startswith("rosettalia") else 0
        entrys.append(entry)
    datafile = {}
    datafile["data"] = entrys
    print(f"{len(entrys)} files found")

    output = open("config.json",'+wt')
    json.dump(datafile, output)
    output.close()

if __name__ == "__main__":
    main()