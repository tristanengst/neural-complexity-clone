import os
import sys


def build_data_dict(dir):
    "Assumes [dir] is a subfolder of where this script is kept"
    collected_data = {}

    for filename in os.listdir(os.getcwd() + "/" + dir + "/"):
        with open(os.getcwd() + "/" + dir + "/" + filename, "r", encoding="utf-8", ) as f:
            data = f.read()
            type_id = filename.index("_") + 1
            dot_idx = filename.index(".")
            type = filename[type_id:dot_idx]

            if type in collected_data:
                collected_data[type] += "\n" + data
            else:
                collected_data[type] = data
            print("Got data from {}".format(filename))

    return collected_data

def create_files(data, dir):
    for k in data:
        filename = os.getcwd() + "/" + dir + "/" + k + ".txt"
        with open(filename, 'w') as f:
            f.write(data[k])


dir = sys.argv[1]
data = build_data_dict(dir)
create_files(data, dir + "_new")
