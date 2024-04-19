import networkx
import csv
import node2vec
from gensim.models import Word2Vec

G = networkx.DiGraph()
with open("data/PeMS08/PEMS08.csv", newline='') as csvfile:
    readers = csv.reader(csvfile)
    next(readers) # 跳过首行
    for row in readers:
        source, target, weight = row
        G.add_edge(int(source), int(target), weight=float(weight))
# 有向图：networkx.DiGraph
# 无向图：networkx.Graph
# 生成有向图
print("Nodes: ", len(G.nodes()))
print("Edges: ", len(G.edges()))
G = node2vec.Graph(G, is_directed=True, p=0.5, q=2)
G.preprocess_transition_probs()
walks = G.simulate_walks(num_walks=100, walk_length=80)

walks = [list(map(str, walk)) for walk in walks]
model = Word2Vec(
    walks,
    vector_size=64,
    window=10,
    min_count=0,
    workers=8,
    sg=1,
    epochs=1000
)
model.wv.save_word2vec_format("data/PeMS08/SE_PeMS08.txt")