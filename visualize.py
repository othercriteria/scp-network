#!/usr/bin/env python

import networkx as nx
import matplotlib.pyplot as plt
import sys
import pickle

source_dir = sys.argv[1]

input = open('%s/parse.pkl' % source_dir, 'rb')
data = pickle.load(input)
tags, scp_tags, links = data['tags'], data['scp_tags'], data['links']

G = nx.DiGraph()
#for scp in scp_tags:
#    G.add_node(scp)
for link in links:
    s = link
    for t in links[link]:
        G.add_edge(s, t)

plt.figure()
nx.draw_graphviz(G)
plt.show()
