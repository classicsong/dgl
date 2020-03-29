import networkx as nx
import csv
from kspath.deviation_path.mps import SingleTargetDeviationPathAlgorithm

m=nx.MultiDiGraph()

entity_map_file = 'entities.tsv'
entity_map = {}

#build name map
with open(entity_map_file, newline='') as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t')
    for row_val in reader:
        name = row_val['name']
        e_id = int(row_val['id'])
        entity_map[e_id] = name

file_path = 'gnbr_triples.tsv' 
edge_rel = {}
#build hetero-graph
with open(file_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t')
    for row_val in reader:
        head = int(row_val['head'])
        tail = int(row_val['tail'])
        rel = row_val['rel']
        m.add_edge(entity_map[head], entity_map[tail], key=rel)
        if edge_rel.get((entity_map[head], entity_map[tail]), None) is None:
            edge_rel[(entity_map[head], entity_map[tail])] = []
        edge_rel[(entity_map[head], entity_map[tail])].append(rel)

print(nx.number_of_edges(m))
target=entity_map[51847]
source=[entity_map[15556], entity_map[14897], entity_map[15587], entity_map[3627]]
# generate topk for each target and source
for src in source:
    paths = []
    maps = SingleTargetDeviationPathAlgorithm.create_from_graph(G=m, target=target, weight=None)
    for path_count, path in enumerate(maps.shortest_simple_paths(source=src)):
        paths.append(path)
        if path_count == 10:
            break
    print("{}->{}:{}".format(target, src, paths))
    for path in paths:
        for i, p in enumerate(path):
            if i == 0:
                continue
            print("{} -> {} -> {}".format(path[i-1], edge_rel[(path[i-1], path[i])], path[i]))

print()
