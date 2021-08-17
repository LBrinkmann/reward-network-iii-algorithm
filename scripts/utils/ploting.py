import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def draw_network(network_json):

    # ToDo: Bidirectional edges  are plotted on top of each other. Plot them in seperate lines.
    # you can add step number as an edge attribute to selected edges. so while plotting, they will also be printed. Think about this.
    # get node information from Node_Trace,origin node and destination node. use add_edge_attributes to add step number.
    options = {
        'node_size': 1000,
        'width': 3,
        'arrowstyle': '-|>',
        'arrowsize': 16,
        'edge_color': 'black'
    }
    a = nx.json_graph.node_link_graph(network_json)
    pos = nx.circular_layout(a)
    nx.draw(a, with_labels=True, node_size=1000, arrow_size=16, pos=pos)
    labels = nx.get_edge_attributes(a, 'weight')
    nx.draw_networkx_edge_labels(a, pos, edge_labels=labels, **options,)
    plt.show()


def changes_hist(df_classes, gamma_s):
    clean = df_classes.replace([np.inf, -np.inf], np.nan).dropna(subset=["Change_L_P", "Change_P_AVP"],
                                                                 how="all")
    ax_list = clean.hist(column=['Change_L_P', 'Change_P_AVP'], bins=30, figsize=(20, 10))
    clean.Change_L_P.plot(kind='kde', ax=ax_list[0][0], secondary_y=True)
    clean.Change_P_AVP.plot(kind='kde', ax=ax_list[0][1], secondary_y=True)
    #ax_list[0][0].set_xlim((clean.Change_L_P.min(), clean.Change_L_P.max()))
    #ax_list[0][1].set_xlim((clean.Change_P_AVP.min(), clean.Change_P_AVP.max()))
    ax_list[0][0].set_title('Change_L_P Histogram Gamma Specific {}'.format(gamma_s))
    ax_list[0][1].set_title('Change_P_AVP Histogram Gamma Specific {}'.format(gamma_s))
    ax_list[0][0].set_xlabel('Change')
    ax_list[0][1].set_xlabel('Change')
    ax_list[0][0].set_ylabel('Freq')
    ax_list[0][1].set_ylabel('Freq')
