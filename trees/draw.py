#!/usr/bin/env python
import ast
import csv
import os.path
import sys

import matplotlib.pyplot as plt
import networkx as nx

from snsim.propagation import read

topic = sys.argv[1]

sub, _ = read.labelled_graph('data/anon_graph_inner_' + topic + '.npz')
sub = nx.from_scipy_sparse_array(sub, create_using=nx.DiGraph())

n = sub.order()

# Set interactive on to draw updates automagically
plt.ion()


############
# SETTINGS #
############


colormap = 'tab20'

base_node_size = 0
tweet_node_size = 180

base_edge_width = 1
tweet_edge_width = 2.5

base_node_edge_color = [0, 0, 0, 0.1]

# Inferred settings
default_node_sizes = [base_node_size] * n
edge_width = [base_edge_width] * sub.number_of_edges()
edge_colors = [[i for i in base_node_edge_color] for x in range(sub.number_of_edges())]
node_colors = [[i for i in base_node_edge_color] for x in range(sub.number_of_nodes())]


############
# PLOTTING #
############


# Create a figure
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)

# Draw graph
def draw_graph():
    plt.cla()
    nx.draw(
        sub,
        node_size=default_node_sizes,
        node_color=node_colors,
        edge_color=edge_colors,
        width=edge_width,
        pos=layout,
        ax=ax,
        arrows=False,
    )


def save_layout(fn):
    with open(fn, "w") as f:
        for k, v in layout.items():
            f.write(f"{k} {v[0]} {v[1]}\n")


def load_layout(fn, whitelist=[]):
    global layout
    layout = {}

    with open(fn) as f:
        for line in f:
            data = line.strip().split()
            if len(whitelist) == 0 or int(data[0]) in whitelist:
                layout[int(data[0])] = [float(data[1]), float(data[2])]


layout_file = f"data/{topic}.layout"
if os.path.exists(layout_file):
    load_layout(layout_file)
else:
    layout = nx.spring_layout(sub)
    save_layout(layout_file)

draw_graph()

################
# TREE DRAWING #
################

trees = []

with open(sys.argv[2], newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=",", quotechar='"')
    next(reader, None)
    for row in reader:
        trees.append(ast.literal_eval(row[3]))

for i, tree in enumerate(trees):
    col = plt.cm.get_cmap(colormap)(i / len(trees))

    edges = []
    for i, j in tree.items():
        if j == -1:
            source = i
        else:
            edges.append((i, j))
    tree = sub.edge_subgraph(edges).reverse()
    nx.draw_networkx_edges(
        tree,
        edge_color=[col] * tree.number_of_edges(),
        width=[tweet_edge_width] * tree.number_of_edges(),
        pos=layout,
        ax=ax,
        node_size=default_node_sizes,
        arrowstyle='->',
        # arrows=False,
    )
    nx.draw_networkx_nodes(
        tree,
        nodelist=[source],
        edgecolors=[col],
        node_color=[(0, 0, 0, 0.0)],
        # node_size=tweet_node_size,
        linewidths=[tweet_edge_width],
        # node_shape="o",
        pos=layout,
        ax=ax,
    )

input("Press Enter to continue...")
