# coding: utf-8
import ast
import csv
import math
import random
import sys
import time

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import scipy

colormap = 'viridis'

# Parse graph
#g = nx.read_adjlist("1k.adjlist")


# Work on a subset cause otherwise it's too large.
#n = 100
#sub = nx.subgraph(g, [str(x) for x in range(n)])

sub = nx.read_graphml(sys.argv[1])

# For drawing just one neighborhood, if not desired set this to False
drawNeighborhood = "0" #sys.argv[3]
if drawNeighborhood:
    neighborhood = list(sub.neighbors(drawNeighborhood)) + [drawNeighborhood]
    # If this is uncommented, will only draw the neighborhood
    #sub = sub.subgraph(neighborhood)

n = sub.order()






# Print sizes of each neighborhood
#for i in sub.nodes():
#    print(i, len(list(sub.neighbors(i))))

# Set interactive on to draw updates automagically
plt.ion()

communities = []


####################
# COMMUNITY LAYOUT #
####################

def bounding_box(points):
    x1 = min(map(lambda x: x[0], points))
    x2 = max(map(lambda x: x[0], points))
    y1 = min(map(lambda x: x[1], points))
    y2 = max(map(lambda x: x[1], points))
    return x1, x2, y1, y2

def center(points):
    x1, x2, y1, y2 = bounding_box(points)
    return x1+abs(x1-x2)/2, y1+abs(y1-y2)/2

def longest_side(points):
    x1, x2, y1, y2 = bounding_box(points)
    return max(map(abs, [x1-x2, y1-y2]))
    
def circle_coords_for(communityId, layout):
    points = list(map(lambda x: layout[x], [str(x) for x in communities.getMembers(communityId)]))
    return center(points), bg_circle_radius_mult * longest_side(points) / 2

# Ordering of communities. Return the position of `coords` which is a node in community id `community`
def do_layout_for(coords, community, ncommunities, nxg, radius_mult):
    theta = 2 * math.pi * community / ncommunities
    radius = len(nxg.nodes()) * radius_mult
    x = radius * math.cos(theta) - radius * math.sin(theta) + coords[0]
    y = radius * math.sin(theta) + radius * math.cos(theta) + coords[1]
    return [x,y]


def clustered_layout(nxg, layouter, radius_mult):
    layout = {}
    ncommunities = float(len(communities.getSubsetIds()))
    community_ids = communities.getSubsetIds()

    #adj_edges = [[0] * len(community_ids) for i in range(len(community_ids))]
    #for id1 in community_ids:
    #    for id2 in community_ids:
    #        if id1 == id2:
    #            continue
    #        for x in communities.getMembers(id1):
    #            for y in communities.getMembers(id2):
    #                if nxg.has_edge(str(x), str(y)):
    #                    adj_edges[id1][id2] += 1


    #ordered_ids = []
    #past_maxes = []
    #while len(ordered_ids) != len(community_ids):
    #    for i in adj_edges:
    #        print(i)
    #    print()
    #    maxes = list(map(max, adj_edges))
    #    next_neighbor = maxes.index(max(maxes))
    #    ordered_ids.append(next_neighbor)
    #    for e in adj_edges:
    #        e[next_neighbor] = 0
    #    adj_edges[next_neighbor] = [0] * len(communities.getSubsetIds())

    #print(ordered_ids)
    #print(community_ids)


    i = 0
    for community in community_ids:
        # Do intra-community layout
        sub = nx.subgraph(nxg, [str(x) for x in communities.getMembers(community)])
        community_layout = layouter(sub)

        # Layout the communities in a circle
        for k, v in community_layout.items():
            layout[k] = do_layout_for(v, community, ncommunities, nxg, radius_mult)

        i = i + 1

    return layout
    

############
# SETTINGS #
############

delta_time = 0.1 # how many seconds to wait between redraws

min_update_frames = 0 # how many frames to draw per simulation step
max_update_frames = 7 # draws uniform randint from that range

sim_params = pd.Series({'edge_probability': 0.045,
                    'discount_factor': 1.,
                    'max_nodes': n,
                    'max_depth': 100,
                    }, dtype=object)


radius_mult = 0.0055
bg_circle_radius_mult = 0.85
layouter = nx.spring_layout

base_node_size = 0
tweet_node_size = 180

base_edge_width = 1
tweet_edge_width = 3.5

base_node_edge_color = [0, 0, 0, 0.1]

shrink_speed = 0.98
decolorize_speed = 0.99

node_labels = False

communities_cmap = plt.get_cmap("Set2")

tweet_cmap = plt.get_cmap("tab10")
n_tweet_colors = 9 # Number of tweet colors to use before reusing previous colors

communityBackgroundPadding = 1.25
communityBackgroundAlpha = 0.2

# Inferred settings
default_node_sizes = [base_node_size]*n
edge_width = [base_edge_width] * sub.number_of_edges()
edge_colors = [[i for i in base_node_edge_color] for x in range(sub.number_of_edges())]
node_colors = [[i for i in base_node_edge_color] for x in range(sub.number_of_nodes())]


############
# PLOTTING #
############


# Create a figure
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)

# Draw graph
def draw_graph():
    global cur_step
    plt.cla()
    cur_step = 0
    nx.draw(sub, node_size=default_node_sizes, node_color=node_colors, edge_color=edge_colors, width=edge_width, pos=layout, ax=ax,with_labels=node_labels)

def save_layout(fn):
    with open(fn, "w") as f:
        for k,v in layout.items():
            f.write(f"{k} {v[0]} {v[1]}\n")
    nk.community.writeCommunities(communities, fn+"c")

def load_layout(fn, whitelist = []):
    global layout
    global communities
    layout = {}
        
    with open(fn) as f:
        for line in f:
            data = line.strip().split()
            if len(whitelist) == 0 or data[0] in whitelist:
                layout[data[0]] = [float(data[1]), float(data[2])]
    draw_graph()

if len(sys.argv) >= 3:
    load_layout(sys.argv[2])#, drawNeighborhood and neighborhood or [])

draw_graph()


def cu(communityId, n=1):
    for i in range(n):
        for node in communities.getMembers(communityId):
            layout[str(node)] = [layout[str(node)][0], layout[str(node)][1] + 0.1]
    draw_graph()
    step()

def cd(communityId, n=1):
    for i in range(n):
        for node in communities.getMembers(communityId):
            layout[str(node)] = [layout[str(node)][0], layout[str(node)][1] - 0.1]
    draw_graph()
    step()

def cl(communityId, n=1):
    for i in range(n):
        for node in communities.getMembers(communityId):
            layout[str(node)] = [layout[str(node)][0] - 0.1, layout[str(node)][1]]
    draw_graph()
    step()

def cr(communityId, n=1):
    for i in range(n):
        for node in communities.getMembers(communityId):
            layout[str(node)] = [layout[str(node)][0] + 0.1, layout[str(node)][1]]
    draw_graph()
    step()

def u(node, n=1):
    for i in range(n):
        layout[str(node)] = [layout[str(node)][0], layout[str(node)][1] + 0.1]
        draw_graph()
    step()

def d(node, n=1):
    for i in range(n):
        layout[str(node)] = [layout[str(node)][0], layout[str(node)][1] - 0.1]
        draw_graph()
    step()

def l(node, n=1):
    for i in range(n):
        layout[str(node)] = [layout[str(node)][0] - 0.1, layout[str(node)][1]]
        draw_graph()
    step()

def r(node, n=1):
    for i in range(n):
        layout[str(node)] = [layout[str(node)][0] + 0.1, layout[str(node)][1]]
        draw_graph()
    step()


def move(node, x, y):
    print(layout[str(node)])
    layout[str(node)] = [x,y]
    draw_graph()
    step()

def redraw():
    layout = clustered_layout(sub, layouter, radius_mult)
    draw_graph()
    step()

# This plots circles around the communities.
def plot_communities():
    communityIds = list(communities.getSubsetIds())
    normalizedIds = plt.Normalize()(communityIds).data
    colormappedIds = communities_cmap(normalizedIds) # get the color of each community
    for i in range(len(communityIds)):
      xy, radius = circle_coords_for(communityIds[i], layout)
      colormappedIds[i][3] = communityBackgroundAlpha # set alpha
      circle = plt.Circle(xy, radius * communityBackgroundPadding, color=colormappedIds[i], zorder=-99)
      ax.add_patch(circle)


########################
# SIMULATION INTERFACE #
########################

def nx_to_scipy_csr(nxg):
    n = nxg.order()
    data = [[0] * n for i in range(n)] # don't do [[0]*n]*n
    for edge in nxg.edges:
        n1 = int(edge[0])
        n2 = int(edge[1])
        data[n1][n2]=1
        data[n2][n1]=1
    return scipy.sparse.csr_matrix(data)

def node_index_of(node_id, g):
    i = 0
    for node in g.nodes:
        if node == str(node_id):
            return i
        i += 1

def edge_index_of(n1, n2, g):
    i = 0
    for edge in g.edges:
        if edge[0] == n1:
            if edge[1] == n2:
                return i

        if edge[0] == n2:
            if edge[1] == n1:
                return i

        i += 1

    return -1

# Get colors of plot
def prepare():
    global edges
    edges = ax.get_children()[1]
    global colors
    colors = edges.get_colors()
    global widths
    widths = edges.get_linewidths()

    global nodes
    nodes = ax.get_children()[0]
    global sizes
    sizes = nodes.get_sizes()
    global ncolors
    ncolors = nodes.get_ec()

# Mutate a color.
def update_edge_color(index, new_value):
    colors[index] = new_value
    edges.set_color(colors)

def update_width(index, new_value):
    widths[index] = new_value
    edges.set_linewidths(widths)
    
def update_node_color(node_id, new_value):
    ncolors[node_index_of(node_id, sub)] = new_value
    nodes.set_ec(ncolors)

def update_size(node_id, new_value):
    sizes[node_index_of(node_id, sub)] = new_value
    nodes.set_sizes(sizes)
    

# Get the color for a given tweet id.
def color_for_tweet(tweet_id, mod):
    normalizedIds = plt.Normalize()([0, math.ceil(tweet_id % (mod+1)), mod]).data
    colormapped = tweet_cmap(normalizedIds) # get the color of each community
    print(colormapped)
    return colormapped[1]


def num_towards_base(i, base, speedFactor):
    if i > base:
        i *= speedFactor
    return i if i > base else base

def list_towards_base(l, base, speedFactor):
    if type(base) == list:
        for i in range(len(l)):
            l[i] = num_towards_base(l[i], base[i], speedFactor)
    else:
        for i in range(len(l)):
            l[i] = num_towards_base(l[i], base, speedFactor)


####################
# SIMULATION SETUP #
####################

    
def draw_step(update):

    tid = update[0]
    n1 = update[1]
    n2 = update[2]

    color = color_for_tweet(tid, n_tweet_colors)
    print(f"tweet {tid}: {n1} {n2}")

    if n1 is not None:
        index = edge_index_of(str(n1), str(n2), sub)
        update_edge_color(index, color)
        update_width(index, tweet_edge_width)
        #update_size(n1, base_node_size)
    else:
        update_size(n2, tweet_node_size)
        update_node_color(n2, color)

    #plt.pause(delta_time)

# For debugging. t=preload() and then step(t.pop())
def preload(n=50):
    tweets = []
    for updates in live_propagate(A, tweet_stream):
        if len(tweets) >= n:
            tweets.reverse()
            return tweets
        tweets.append(updates)

def run():
    global shrink_speed
    global decolorize_speed
    for updates in live_propagate(A, tweet_stream):
            random.shuffle(updates)
            for update in updates:
                frames_till_step = round((random.uniform(min_update_frames, max_update_frames)) / math.sqrt(math.sqrt(math.sqrt(len(updates)))))
                for i in range(frames_till_step):
                    list_towards_base(sizes, base_node_size, shrink_speed)
                    nodes.set_sizes(sizes)

                    list_towards_base(widths, base_edge_width, shrink_speed)
                    edges.set_linewidths(widths)

                    for c in colors:
                        list_towards_base(c, base_node_edge_color, decolorize_speed)
                    edges.set_color(colors)

                    for c in ncolors:
                        list_towards_base(c, base_node_edge_color, decolorize_speed)
                    nodes.set_ec(ncolors)

                    plt.pause(delta_time)

                if time.time() - start_time < 35:
                    draw_step(update)
                else:
                    shrink_speed *= 0.99
                    decolorize_speed *= 0.99

cur_step = 0

def step():
    global cur_step
    global start_time
    if cur_step == 0:
        plot_communities()
    elif cur_step == 1:
        prepare()
        start_time = time.time()
        time.sleep(1)
        run()
    cur_step += 1


#
################
# TREE DRAWING #
################

trees = []

with open(sys.argv[3], newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=",", quotechar='"')
    next(reader, None)
    for row in reader:
        trees.append(ast.literal_eval(row[3]))

prepare()
for ti, tree in enumerate(trees):
    col = plt.cm.get_cmap(colormap)(ti/len(trees))
    for i,j in tree.items():
        if j == -1:
            update_node_color(str(i), col)
            update_size(str(i), tweet_node_size)
        index = edge_index_of(f'{i}', f'{j}', sub)
        if index != -1:
            update_edge_color(index, col)
            update_width(index, 3)

