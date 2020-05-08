import sys
import os
import random

dir = sys.argv[1]
inputs = sys.argv[2:]
num_sentences = 10000000
sentences = {}
output_name = ""

for input in inputs:
    filename = os.getcwd() + "/" + dir + "/" + input
    output_name += input[:input.index(".")] + "_"

    with open(filename, "r") as f:
        data = f.read()
        data_as_list = data.split("\n")
        dot_idx = input.index(".")
        cat = input[:dot_idx]

        sentences[cat] = data_as_list
        num_sentences = min(num_sentences, len(data_as_list))
        print("Got data from {}".format(input))

# Randomly sort each list within sentences
for k in sentences:
    sentences[k] = random.sample(sentences[k], len(sentences[k]))


result = []
for i in range(num_sentences):
    for k in sentences:
        result.append(sentences[k][i])

str_result = ""
for r in result:
    str_result += r + "\n"

output_name = output_name[:-1] + ".txt"
print(output_name)

with open(output_name, "w+") as f:
    f.write(str_result)
