import os
import argparse
import numpy as np

class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()
        self.add_argument('--file', type=str, help='file to split')
        self.add_argument('--train_rate', type=float, default=0.9)
        self.add_argument('--valid_rate', type=float, default=0.05)

def run(args):
    lines = []
    with open(args.file) as f:
        for line in f:
            lines.append(line)

    num_triples = len(lines)
    num_train = int(num_triples * args.train_rate)
    num_valid = int(num_triples * args.valid_rate)
    num_test = num_triples - num_train - num_valid

    print(num_train)
    print(num_valid)
    print(num_test)

    seed = np.arange(num_triples)
    np.random.shuffle(seed)
    seed = seed.tolist()

    train_lines = []
    valid_lines = []
    test_lines = []
    for i in range(num_triples):
        idx = seed[i]
        h, r, t = lines[idx].strip().split('\t')
        if int(r) != 6:
            train_lines.append(lines[idx])
        elif idx < num_train:
            train_lines.append(lines[idx])
        elif idx < num_train + num_valid:
            valid_lines.append(lines[idx])
        else:
            test_lines.append(lines[idx])

    with open("train.tsv", "w+") as f:
        f.writelines(train_lines)

    with open("valid.tsv", "w+") as f:
        f.writelines(valid_lines)

    with open("test.tsv", "w+") as f:
        f.writelines(test_lines)


if __name__ == '__main__':
    args = ArgParser().parse_args()
    run(args)
