import os
import argparse

class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()
        self.add_argument('--entity', type=str)
        self.add_argument('--relation', type=str)
        self.add_argument('--triple', type=str)
        self.add_argument('--selected_rels', type=int, nargs='+')

def get_id(dict, key):
    id = dict.get(key, None)
    if id is None:
        id = len(dict)
        dict[key] = id
    return id

def run(args):
    entities = []
    with open(args.entity) as f_ent:
        for line in f_ent:
            entity, id = line.strip().split('\t')
            entities.append(entity)

    relations = []
    with open(args.relation) as f_rel:
        for line in f_rel:
            relation, id = line.strip().split('\t')
            relations.append(relation)

    selected_rels = args.selected_rels
    lines = []
    with open(args.triple) as f_tri:
        for line in f_tri:
            h, r, t = line.strip().split('\t')
            if int(r) in selected_rels:
                lines.append((entities[int(h)], 
                             relations[int(r)],
                             entities[int(t)]))

    # generate cleaned dataset
    entity_dict = {}
    rel_dict = {}
    triples = []
    for line in lines:
        h, r, t = line
        hid = get_id(entity_dict, h)
        rid = get_id(rel_dict, r)
        tid = get_id(entity_dict, t)

        triples.append("{}\t{}\t{}\n".format(hid, rid, tid))

    entities = []
    for key, val in sorted(entity_dict.items(), key=lambda item: item[1]):
        entities.append("{}\t{}\n".format(key, val))

    relations = []
    for key, val in sorted(rel_dict.items(), key=lambda item: item[1]):
        relations.append("{}\t{}\n".format(key, val))

    with open("entities.tsv", "w+") as f:
        f.writelines(entities)
    with open("relations.tsv", "w+") as f:
        f.writelines(relations)
    with open("triples.tsv", "w+") as f:
        f.writelines(triples)

if __name__ == '__main__':
    args = ArgParser().parse_args()
    run(args)