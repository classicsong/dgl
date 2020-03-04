import os
import argparse
import numpy as np

class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()
        self.add_argument('--src_drug', type=str)
        self.add_argument('--id_map', type=str)
        self.add_argument('--src_target', type=str)
        self.add_argument('--target_map', type=str)

def handle_drug(args):
    target_drug = []
    with open(args.src_drug) as fs:
        for line in fs:
            entity, _ = line.strip().split('\t')
            target_drug.append(entity)

    id_map = {}
    with open(args.id_map) as fd:
        for line in fd:
            entity, id = line.strip().split('\t')
            if entity.startswith("<http://bio2rdf.org/drugbank:"):
                id_map[entity[29:36]] = id

    lines = []
    skip = 0
    for target in target_drug:
        if id_map.get(target, None) is None:
            skip += 1
            continue
        lines.append("{}\t{}\n".format(target, id_map[target]))
    print("skip {}/{}".format(skip, len(lines)))

    with open("drug_entity.tsv", "w+") as f:
        f.writelines(lines)

def handle_target(args):
    target_id = []
    with open(args.src_target) as fs:
        for line in fs:
            _, id = line.strip().split('\t')
            target_id.append(int(id))
    target_id = np.array(target_id)
    target_id = np.unique(target_id)
    target_id = target_id.tolist()

    target_entities = []
    with open(args.target_map) as fd:
        for line in fd:
            entity, id = line.strip().split('\t')
            if int(id) in target_id:
                target_entities.append(entity)

    lines = []
    with open(args.id_map) as f:
        for line in f:
            entity, id = line.strip().split('\t')
            if entity in target_entities:
                lines.append("{}\t{}\n".format(entity, id))

    with open("target_entity.tsv", "w+") as f:
        f.writelines(lines)

def run(args):
    handle_drug(args)
    handle_target(args)


if __name__ == '__main__':
    args = ArgParser().parse_args()
    run(args)