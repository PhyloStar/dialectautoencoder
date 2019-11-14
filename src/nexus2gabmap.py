#!/usr/bin/env python3

import sys
import pandas as pd

def read_nexus_diff(filename):
    labels = []
    data = []
    with open(filename, "r") as fp:
        line = fp.readline()
        while not line.startswith("BEGIN Distances;"):
            line = fp.readline()
        while not line.startswith("MATRIX"):
            line = fp.readline()
        line = fp.readline()
        while line.startswith("[1]"):
            line = line.strip().split("\t")
            labels.append(line[1].strip("'").strip('"'))
            data.append([float(x) for x in line[2:]])
            line = fp.readline()
    return (data, labels)

def write_gabmap_diff(data, labels, filename):
    with open(filename, "w") as fp:
        for label in labels:
            print("\t{}".format(label), end="", file=fp)
        print(file=fp)
        for line_num, row in enumerate(data):
            print("{}".format(labels[line_num]), end="", file=fp)
            for i, x in enumerate(row):
                if x == 0.0: # gabmap does not like 0.0
                    print("\t0", end="", file=fp)
                elif x < 0.0:
                    print("Warning: negative value ({}), writing as 0.0".format(x))
                    print("\t0.0", end="", file=fp)
                else:
                    print("\t{}".format(row[i]), end="", file=fp)
            print(file=fp)


d, l = read_nexus_diff(sys.argv[1])

dd = pd.DataFrame(d)
dd.index = l
dd.columns = l

# for i in range(len(l)):
#     for j in range(i+1, len(l)):
#         print(i, j, end=": ")
#         print(dd.iloc[i,j] - dd.iloc[j,i])

write_gabmap_diff(d, l, sys.argv[1].replace(".nex", ".differences"))


