# front-door
import numpy as np
import networkx as nx
import graphviz
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

import matplotlib as mpl

mpl_backend = mpl.get_backend()
if mpl_backend != "Qt5Agg":
    mpl.use("Qt5Agg")
else:
    pass

adj_mat = np.array([
    [0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
])
graph = nx.DiGraph(adj_mat)

num_nodes = adj_mat.shape[0]
vars = ['U', 'X', 'Y', 'Z', 'U_x', 'U_y', 'U_z']
var_names = ['Motivation', 'GPS usage', 'Spatial memory', 'Hippocampus volume', '', '', '']
for i in range(num_nodes):
    graph.nodes[i]['var'] = vars[i]
    graph.nodes[i]['var_name'] = var_names[i]
print(graph.nodes.data())

label_dict = {0: 'U', 1: 'X', 2: 'Y', 3: 'Z', 4: 'U_x', 5: 'U_y', 6: 'U_z'}
graph = nx.relabel_nodes(graph, label_dict)

options = {
    'node_color': 'grey',
    'node_size': 1000,
    'width': 1,
}
nx.draw(graph, with_labels=True, font_weight='bold', pos=nx.planar_layout(graph), **options)

# alternatively
graph = graphviz.Digraph(format='png', engine='neato')
nodes = ['U: Motivation', 'X: GPS usage', 'Z: Hippocampus volume', 'Y: Spatial memory', 'U_x', 'U_z', 'u_y']


class GPSMemorySCM:
    def __init__(self, random_seed=None):
        self.random_seed = random_seed
        self.u_x = stats.truncnorm(0, np.infty, scale=5)
        self.u_y = stats.norm(scale=2)
        self.u_z = stats.norm(scale=2)
        self.u = stats.truncnorm(0, np.infty, scale=4)

    def sample(self, sample_size=100, treatment_value=None):
        """Samples from the SCM"""
        if self.random_seed:
            np.random.seed(self.random_seed)

        u_x = self.u_x.rvs(sample_size)
        u_y = self.u_y.rvs(sample_size)
        u_z = self.u_z.rvs(sample_size)
        u = self.u.rvs(sample_size)

        if treatment_value:
            gps = np.array([treatment_value] * sample_size)
        else:
            gps = u_x + 0.7 * u

        hippocampus = -0.6 * gps + 0.25 * u_z
        memory = 0.7 * hippocampus + 0.25 * u

        return gps, hippocampus, memory

    def intervene(self, treatment_value, sample_size=100):
        """Intervenes on the SCM"""
        return self.sample(treatment_value=treatment_value, sample_size=sample_size)


scm = GPSMemorySCM()
gps_obs, hippocampus_obs, memory_obs = scm.sample(100)

treatments = []
exp_results = []
for treatment in np.arange(1, 21):
    gps_hours, hippocampus, memory = scm.intervene(treatment_value=treatment, sample_size=30)
    exp_results.append(memory)
    treatments.append(gps_hours)

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
axs[0].scatter(gps_obs, memory_obs, alpha=0.2)
axs[0].set_xlabel('GPS usage')
axs[0].set_ylabel('Spatial memory change')
axs[0].set_title('Observational')
axs[1].scatter(gps_hours, memory, alpha=0.2)
axs[0].set_title('Interventional')
fig.tight_layout()


for idx, ax in enumerate(axs.flat):
    ax.scatter(xvars[idx], yvars[idx], alpha=0.2)
    ax.set_xlabel(f'{xlabs[idx]}')
    ax.set_ylabel(f'{ylabs[idx]}')
plt.suptitle(title)
fig.tight_layout()
