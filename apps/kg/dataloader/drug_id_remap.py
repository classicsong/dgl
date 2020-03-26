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
    id_rev_map = {}
    with open(args.id_map) as fd:
        for line in fd:
            entity, id = line.strip().split('\t')
            id_map[entity] = id
            id_rev_map[id] = entity

    drug_ids = []
    skip = 0
    for target in target_drug:
        if id_map.get(target, None) is None:
            skip += 1
            continue
        drug_ids.append(id_map[target])
    print("skip {}".format(skip))
    drug_ids = np.array(drug_ids)
    drug_ids = np.unique(drug_ids)
    drug_ids = drug_ids.tolist()

    lines = []
    for drug in drug_ids:
        lines.append("{}\t{}\n".format(id_rev_map[drug], drug))
    with open("drug_entity.tsv", "w+") as f:
        f.writelines(lines)

def handle_target(args):
    target_target = []
    with open(args.src_target) as fs:
        for line in fs:
            _, entity = line.strip().split('\t')
            #data = line.strip().split('\t')
            #entity = data[0]
            target_target.append(entity)

    
    target_map = {}
    target_rev_map = {}
    with open(args.target_map) as fd:
        for line in fd:
            entity, id = line.strip().split('\t')
            target_map[entity] = id
            target_rev_map[id] = entity
    
    target_ids = []
    skip = 0
    for target in target_target:
        if target_map.get(target, None) is None:
            skip += 1
            continue
        target_ids.append(target_map[target])
    print("skip {}".format(skip))
    target_ids = np.array(target_ids)
    target_ids = np.unique(target_ids)
    target_ids = target_ids.tolist()
    lines = []
    for target in target_ids:
        lines.append("{}\t{}\n".format(target_rev_map[target], target))

    with open("target_entity.tsv", "w+") as f:
        f.writelines(lines)

def run(args):
    handle_drug(args)
    handle_target(args)


if __name__ == '__main__':
    args = ArgParser().parse_args()
    run(args)
