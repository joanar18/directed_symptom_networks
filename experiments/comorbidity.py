
# Modules --------------------------------------------------------------------------------------
import pandas as pd
import networkx as nx
from GraphicalModels.utils.data_utils  import VARIABLE_DEFINITIONS, DISORDER_VARIABLES
from collections import defaultdict
import matplotlib.pyplot as plt

#from models import DirectedInfluenceIsingModel
#from experiment import get_data


# Data ------------------------------------------------------------------------------------------
G = nx.read_gml('comorbidity.gml')

# Variable-to-disorder map --------------------------------------------------------------------
disorder_map = DISORDER_VARIABLES
symptom_to_disorder = {symptom: disorder for disorder, symptoms in disorder_map.items() for symptom in symptoms}

dupes = (pd.Series(list(symptom_to_disorder.keys())).value_counts()[lambda s: s > 1])
if len(dupes):
    raise ValueError(
        f"Duplicate symptom labels across disorders: {', '.join(dupes.index.tolist())}"
    )

for node in G.nodes():
    G.nodes[node]["disorder"] = symptom_to_disorder.get(node, "unknown")

# Bridging -----------------------------------------------------------------------------------------------
bridging_stats = defaultdict(lambda: {
    'pos_out': 0, 'neg_out': 0,
    'pos_in': 0, 'neg_in': 0,
    'total_cross_disorder': 0
})

for u, v, d in G.edges(data=True):
    group_u = G.nodes[u].get('disorder', 'unknown')
    group_v = G.nodes[v].get('disorder', 'unknown')
    weight = d.get('weight', 0)

    if group_u != group_v and 'unknown' not in (group_u, group_v):
        if weight > 0:
            bridging_stats[u]['pos_out'] += 1
            bridging_stats[v]['pos_in'] += 1
        elif weight < 0:
            bridging_stats[u]['neg_out'] += 1
            bridging_stats[v]['neg_in'] += 1

        bridging_stats[u]['total_cross_disorder'] += 1
        bridging_stats[v]['total_cross_disorder'] += 1

bridge_df = pd.DataFrame.from_dict(bridging_stats, orient='index')
bridge_df['disorder'] = [G.nodes[node].get('disorder', 'unknown') for node in bridge_df.index]
bridge_df.sort_values('total_cross_disorder', ascending=False, inplace=True)
bridge_df.to_csv('bridge_symptoms.csv')
top = bridge_df.head(15)
print(top)
#bridge_df.to_csv('bridge_summary.csv')


# group-to-group edge accumulators
group_edges = defaultdict(lambda: {'positive': 0.0, 'negative': 0.0})

for u, v, d in G.edges(data=True):
    disorder_u = G.nodes[u].get('disorder', 'unknown')
    disorder_v = G.nodes[v].get('disorder', 'unknown')
    weight = d.get('weight', 0.0)

    if disorder_u != disorder_v and 'unknown' not in (disorder_u, disorder_v):
        if weight > 0:
            group_edges[(disorder_u, disorder_v)]['positive'] += weight
        elif weight < 0:
            group_edges[(disorder_u, disorder_v)]['negative'] += weight

G_summary = nx.DiGraph()

# Add nodes (disorders)
disorder_names = set([G.nodes[n].get('disorder') for n in G.nodes()])
for disorder in disorder_names:
    if disorder != 'unknown':
        G_summary.add_node(disorder)

# Add weighted edges
for (src, tgt), values in group_edges.items():
    pos_w = values['positive']
    neg_w = values['negative']
    net_w = pos_w + neg_w  # negative is already < 0

    if abs(net_w) > 0.01:  # ignore very weak total weights
        G_summary.add_edge(src, tgt, weight=net_w, pos=pos_w, neg=neg_w)

# Create a summary DataFrame for inspection
import pandas as pd
summary_data = [
    {'from': u, 'to': v, 'net_weight': d['weight'], 'positive': d['pos'], 'negative': d['neg']}
    for u, v, d in G_summary.edges(data=True)
]

summary_df = pd.DataFrame(summary_data)
summary_df.sort_values(by='net_weight', ascending=False, inplace=True)
print(summary_df)
summary_df.to_csv('bridging_comorbidity.csv')

# Plot 
G_summary = nx.DiGraph()

# Add nodes (disorders)
disorders = set(summary_df['from']).union(set(summary_df['to']))
for disorder in disorders:
    G_summary.add_node(disorder)

for _, row in summary_df.iterrows():
    G_summary.add_edge(
        row['from'],
        row['to'],
        weight=row['net_weight'],
        color='green' if row['net_weight'] > 0 else 'red',
        width=abs(row['net_weight'])
    )

nx.write_gml(G_summary, "inter_disorder_network.gml")
pos = nx.spring_layout(G_summary, seed=42, k=1.2)
edges = G_summary.edges(data=True) 
colors = [d['color'] for _, _, d in edges]
widths = [d['width'] for _, _, d in edges]

plt.figure(figsize=(8, 6))
nx.draw_networkx_nodes(G_summary, pos, node_color='lightgray', node_size=400)
nx.draw_networkx_labels(G_summary, pos, font_size=7, font_weight='bold')
nx.draw_networkx_edges(G_summary, pos, edge_color=colors, width=widths, arrows=True, arrowstyle='->', connectionstyle='arc3,rad=0.1')
plt.title('Inter-Disorder Influence Network (Signed)', fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.show()


nx.set_node_attributes(G, symptom_to_disorder, name="disorder")
nx.write_gml(G, "comorbidity.gml")