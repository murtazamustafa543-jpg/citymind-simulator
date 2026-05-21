"""
challenge5crime.py
CityMind — Challenge 5: Crime Risk Prediction and Integration
"""

import random
import math
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

try:
    from city_graph import CityGraph, NodeType, CrimeRisk
    _STANDALONE = False
except ImportError:
    _STANDALONE = True


class CrimeRiskPipeline:

    def __init__(self, graph: CityGraph,
                 n_clusters:    int   = 3,
                 k_neighbours:  int   = 5,
                 noise_rate:    float = 0.10,
                 seed:          int   = None):
        self.graph        = graph
        self.n_clusters   = n_clusters
        self.k_neighbours = k_neighbours
        self.noise_rate   = noise_rate
        self.seed         = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.features:      np.ndarray = None  
        self.node_ids:      list       = None  
        self.cluster_labels: np.ndarray = None 
        self.risk_map:      dict       = {}    

        self._indus_proximity: dict = {}
        self._precompute_proximity()


    def _precompute_proximity(self):
        from collections import deque

        industrial_nodes = [
            nid for nid, node in self.graph.nodes.items()
            if node.location_type == NodeType.INDUSTRIAL
        ]

        if not industrial_nodes:
            for nid in self.graph.nodes:
                self._indus_proximity[nid] = 0.0
            return

        dist = {nid: float('inf') for nid in self.graph.nodes}
        queue = deque()
        for ind in industrial_nodes:
            dist[ind] = 0
            queue.append((ind, 0))

        while queue:
            current, d = queue.popleft()
            for nb in self.graph.adjacency[current]:
                edge = self.graph.get_edge(current, nb)
                if edge and edge.is_blocked:
                    continue
                if dist[nb] == float('inf'):
                    dist[nb] = d + 1
                    queue.append((nb, d + 1))

        for nid, d in dist.items():
            self._indus_proximity[nid] = 1.0 / (d + 1)


    def _build_features(self) -> np.ndarray:
        node_ids = list(self.graph.nodes.keys())
        X = []

        max_density = max(
            (n.population_density for n in self.graph.nodes.values()),
            default=1.0
        ) or 1.0

        safe_types = {NodeType.HOSPITAL, NodeType.SCHOOL}

        for nid in node_ids:
            node = self.graph.nodes[nid]
            nt   = node.location_type

            f1 = node.population_density / max_density

            f2 = self._indus_proximity.get(nid, 0.0)

            f3 = 1.0 if nt == NodeType.INDUSTRIAL else 0.0

            f4 = 1.0 if nt == NodeType.RESIDENTIAL else 0.0

            f5 = 0.0
            for nb in self.graph.adjacency[nid]:
                if self.graph.nodes[nb].location_type in safe_types:
                    f5 = 1.0
                    break

            X.append([f1, f2, f3, f4, f5])

        self.node_ids = node_ids
        return np.array(X, dtype=float)


    def _kmeans_cluster(self, X_scaled: np.ndarray) -> np.ndarray:
       
        print(f"[C5] Step 1: K-Means clustering  (k={self.n_clusters})...")

        km = KMeans(
            n_clusters=self.n_clusters,
            n_init=10,
            random_state=self.seed,
        )
        labels = km.fit_predict(X_scaled)

        
        centres = km.cluster_centers_   
        risk_scores = centres[:, 0] + centres[:, 1] + centres[:, 2]
        rank = np.argsort(risk_scores)  


        cluster_to_risk = {}
        risk_levels = [CrimeRisk.LOW, CrimeRisk.MEDIUM, CrimeRisk.HIGH]
        for tier, cluster_id in enumerate(rank):
            cluster_to_risk[cluster_id] = risk_levels[tier]

        for cid in range(self.n_clusters):
            count = np.sum(labels == cid)
            risk  = cluster_to_risk[cid]
            print(f"    Cluster {cid}: {count:3d} nodes → {risk.value} risk")

        self.cluster_to_risk = cluster_to_risk
        return labels


    def _generate_synthetic_dataset(self) -> tuple[np.ndarray, list]:
      
        print(f"[C5] Step 2: Generating synthetic crime dataset...")

        labels     = []
        max_density = max(
            (n.population_density for n in self.graph.nodes.values()),
            default=1.0
        ) or 1.0
        high_density_threshold = 0.6 * max_density

        safe_types = {NodeType.HOSPITAL, NodeType.SCHOOL}
        low_types  = {NodeType.POWER_PLANT, NodeType.AMBULANCE_DEPOT}

        for nid in self.node_ids:
            node = self.graph.nodes[nid]
            nt   = node.location_type
            prox = self._indus_proximity.get(nid, 0.0)

            if nt == NodeType.INDUSTRIAL:
                if (node.population_density >= high_density_threshold
                        or prox >= 0.5):
                    label = CrimeRisk.HIGH
                else:
                    label = CrimeRisk.MEDIUM

            elif nt == NodeType.RESIDENTIAL:
                adj_to_industrial = any(
                    self.graph.nodes[nb].location_type == NodeType.INDUSTRIAL
                    for nb in self.graph.adjacency[nid]
                )
                adj_to_safe = any(
                    self.graph.nodes[nb].location_type in safe_types
                    for nb in self.graph.adjacency[nid]
                )
                if adj_to_industrial:
                    label = CrimeRisk.MEDIUM
                elif adj_to_safe:
                    label = CrimeRisk.LOW
                else:
                    label = CrimeRisk.LOW

            elif nt in low_types:
                label = CrimeRisk.LOW

            elif nt in safe_types:
                label = CrimeRisk.LOW

            else:
                label = CrimeRisk.LOW

            labels.append(label)

        n_noise = int(len(labels) * self.noise_rate)
        noise_indices = random.sample(range(len(labels)), n_noise)
        all_risks = [CrimeRisk.LOW, CrimeRisk.MEDIUM, CrimeRisk.HIGH]
        for idx in noise_indices:
            current = labels[idx]
            options = [r for r in all_risks if r != current]
            labels[idx] = random.choice(options)

        counts = defaultdict(int)
        for l in labels:
            counts[l] += 1
        print(f"    Dataset: HIGH={counts[CrimeRisk.HIGH]}  "
              f"MEDIUM={counts[CrimeRisk.MEDIUM]}  LOW={counts[CrimeRisk.LOW]}  "
              f"(+{n_noise} noise flips)")

        return labels

    def _train_knn(self, X_scaled: np.ndarray,
                   synthetic_labels: list) -> list:
        print(f"[C5] Step 2: Training KNN classifier  (k={self.k_neighbours})...")

        label_to_int = {CrimeRisk.LOW: 0, CrimeRisk.MEDIUM: 1, CrimeRisk.HIGH: 2}
        int_to_label = {0: CrimeRisk.LOW, 1: CrimeRisk.MEDIUM, 2: CrimeRisk.HIGH}

        y = np.array([label_to_int[l] for l in synthetic_labels])

        knn = KNeighborsClassifier(
            n_neighbors=self.k_neighbours,
            metric='euclidean',
            weights='distance',  
        )
        knn.fit(X_scaled, y)

        y_pred = knn.predict(X_scaled)
        predicted = [int_to_label[p] for p in y_pred]

        accuracy = np.mean(y_pred == y)
        print(f"    KNN fit accuracy (vs synthetic labels): {accuracy*100:.1f}%")

        counts = defaultdict(int)
        for p in predicted:
            counts[p] += 1
        print(f"    Predictions: HIGH={counts[CrimeRisk.HIGH]}  "
              f"MEDIUM={counts[CrimeRisk.MEDIUM]}  LOW={counts[CrimeRisk.LOW]}")

        return predicted


    def _apply_to_graph(self, knn_predictions: list):
        print(f"[C5] Step 3: Writing risk levels to shared graph...")

        risk_map = {
            self.node_ids[i]: knn_predictions[i]
            for i in range(len(self.node_ids))
        }
        self.risk_map = risk_map
        self.graph.apply_all_crime_risks(risk_map)

        multiplier_map = {
            CrimeRisk.HIGH: 1.5,
            CrimeRisk.MEDIUM: 1.2,
            CrimeRisk.LOW: 1.0,
        }
        sample_edges = list(self.graph.edges.values())[:3]
        print(f"    Sample edge costs after update:")
        for edge in sample_edges:
            a, b = edge.node_a, edge.node_b
            ra = self.graph.nodes[a].crime_risk_level.value
            rb = self.graph.nodes[b].crime_risk_level.value
            print(f"      {a}↔{b}  risk=({ra}/{rb})  "
                  f"effective_cost={edge.effective_cost:.2f}")


    def _deploy_police(self, num_officers: int = 10) -> dict:
      
        print(f"[C5] Step 4: Deploying {num_officers} police officers...")

        high_nodes   = sorted(
            [nid for nid, n in self.graph.nodes.items()
             if n.crime_risk_level == CrimeRisk.HIGH],
            key=lambda nid: self.graph.nodes[nid].population_density,
            reverse=True,
        )
        medium_nodes = sorted(
            [nid for nid, n in self.graph.nodes.items()
             if n.crime_risk_level == CrimeRisk.MEDIUM],
            key=lambda nid: self.graph.nodes[nid].population_density,
            reverse=True,
        )

        if not high_nodes and not medium_nodes:
            print("    [WARNING] No HIGH or MEDIUM risk nodes — no officers deployed.")
            return {}

        weighted = (high_nodes * 3) + medium_nodes

        for node in self.graph.nodes.values():
            node.police_officers = 0

        deployment: dict = defaultdict(int)

        for i in range(num_officers):
            target = weighted[i % len(weighted)]
            deployment[target]          += 1
            self.graph.nodes[target].police_officers += 1

        print(f"    Officers placed across {len(deployment)} nodes:")
        for nid, count in sorted(deployment.items(),
                                  key=lambda x: -x[1]):
            risk = self.graph.nodes[nid].crime_risk_level.value
            dens = self.graph.nodes[nid].population_density
            print(f"      Node {nid}  [{risk:<6}]  density={dens:<4.0f}  "
                  f"officers={count}")

        total_deployed = sum(deployment.values())
        print(f"    Total deployed: {total_deployed} / {num_officers}")
        return dict(deployment)


    def run(self, num_officers: int = 10) -> dict:
        print("=" * 60)
        print("  Challenge 5 — Crime Risk Prediction & Police Deployment")
        print("  Algorithm: K-Means (unsupervised) + KNN (supervised)")
        print("=" * 60)

        print(f"[C5] Building feature matrix for {len(self.graph.nodes)} nodes...")
        X = self._build_features()
        self.features = X

        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print(f"    Feature matrix shape: {X.shape}  (nodes × features)")
        print(f"    Features: density, indus_proximity, is_industrial, "
              f"is_residential, near_safe\n")

        cluster_labels = self._kmeans_cluster(X_scaled)
        self.cluster_labels = cluster_labels

        cluster_col = cluster_labels.reshape(-1, 1).astype(float)
        X_scaled_with_cluster = np.hstack([X_scaled, cluster_col])
        print()

        synthetic_labels = self._generate_synthetic_dataset()
        knn_predictions  = self._train_knn(X_scaled_with_cluster, synthetic_labels)
        print()

        self._apply_to_graph(knn_predictions)
        print()

        deployment = self._deploy_police(num_officers)
        self.deployment = deployment
        print()

        self._print_risk_grid()

        return {
            "risk_map"   : self.risk_map,
            "deployment" : deployment,
        }


    def _print_risk_grid(self):
        symbols = {
            CrimeRisk.HIGH:   " ! ",
            CrimeRisk.MEDIUM: " ~ ",
            CrimeRisk.LOW:    " . ",
        }
        size = self.graph.grid_size

        print("Crime Risk Grid  (!=High  ~=Medium  .=Low)")
        print("    " + "".join(f"{c:^3}" for c in range(size)))
        print("   +" + "---" * size + "+")
        for r in range(size):
            row = f"{r:2} |"
            for c in range(size):
                risk = self.graph.nodes[(r,c)].crime_risk_level
                row += symbols[risk]
            print(row + "|")
        print("   +" + "---" * size + "+\n")

        counts = defaultdict(int)
        for node in self.graph.nodes.values():
            counts[node.crime_risk_level] += 1
        print("Risk level summary:")
        print(f"  HIGH   (1.5x cost): {counts[CrimeRisk.HIGH]:3d} nodes")
        print(f"  MEDIUM (1.2x cost): {counts[CrimeRisk.MEDIUM]:3d} nodes")
        print(f"  LOW    (1.0x cost): {counts[CrimeRisk.LOW]:3d} nodes")
        print()


if __name__ == "__main__":
    from city_graph import CityGraph, NodeType, CrimeRisk
    from challenge1csp import CityLayoutCSP

    graph = CityGraph(grid_size=10)
    csp   = CityLayoutCSP(graph, seed=42)
    csp.solve()
    print()

    pipeline = CrimeRiskPipeline(graph, seed=42)
    result   = pipeline.run()
    risk_map   = result["risk_map"]
    deployment = result["deployment"]

    print("=" * 60)
    print(f"  Done. {len(risk_map)} nodes assigned crime risk levels.")
    print(f"  Edge costs updated in shared graph.")
    print(f"  Police deployed to {len(deployment)} nodes.")
    print("=" * 60)

    print("\nSample node risks:")
    for nid, node in list(graph.nodes.items())[::15]:
        officers = node.police_officers
        print(f"  {node}  risk={node.crime_risk_level.value}  "
              f"multiplier={node.crime_multiplier}x  officers={officers}")