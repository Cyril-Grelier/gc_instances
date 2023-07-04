"""
Reduce instance and create an animation of the reduction

work only for the instance 0_test created for the animation
you will need to change layouts and colors of the cliques for other instances
"""
from __future__ import annotations
from dataclasses import dataclass, field
import os
import time
import codecs

import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt

sns.set_theme()


def to_vertex(vertex: int) -> str:
    """Convert vertex number to vertex name"""
    if vertex < 10:
        return "v" + chr(0x2080 + vertex)
    dec = vertex // 10
    uni = vertex % 10
    return "v" + chr(0x2080 + dec) + chr(0x2080 + uni)


@dataclass
class Node:
    """Representation of a Node for graph"""

    old_number: int = -1
    new_number: int = -1
    weight: int = -1
    neighbors_int: list[int] = field(default_factory=list)
    neighbors_nodes: list[Node] = field(default_factory=list)
    reduced: bool = False


def read_col_files(instance_file: str) -> tuple[int, list[tuple[int, int]]]:
    """Read .col (DIMACS) file"""
    edges_list: list[tuple[int, int]] = []
    with open(instance_file, "r", encoding="utf8") as file:
        for line_ in file:
            line = line_.strip()
            if line.startswith("c"):
                continue
            if line.startswith("p"):
                _, _, nb_vertices_str, _ = line.split()
                nb_vertices = int(nb_vertices_str)
            elif line.startswith("e"):
                _, vertex1_str, vertex2_str = line.split()
                vertex1_ = int(vertex1_str) - 1
                vertex2_ = int(vertex2_str) - 1
                if vertex1_ == vertex2_:
                    continue
                vertex1 = min(vertex1_, vertex2_)
                vertex2 = max(vertex1_, vertex2_)
                edges_list.append((vertex1, vertex2))
    return nb_vertices, edges_list


def read_weights_file(weights_file: str, nb_vertices: int) -> list[int]:
    """Read weights file and check the number of vertices"""
    if weights_file == "":
        return [1] * nb_vertices

    with open(weights_file, "r", encoding="utf8") as file:
        weights = list(map(int, file.readlines()))
    assert len(weights) == nb_vertices
    return weights


class Graph:
    """Representation of a graph"""

    def __init__(self, instance_file: str, weights_file: str) -> None:
        """Load graph from file

        Args:
            instance_file (str): file containing the instance (.col file)
            weights_file (str): file containing the weights of the instance (col.w file)
        """

        self.name: str
        self.nb_vertices: int
        self.nb_edges: int
        self.edges_list: list[tuple[int, int]]
        self.adjacency_matrix: list[list[bool]]
        self.neighborhood: list[list[int]]
        self.weights: list[int]
        self.reduced_vertices: list[int]
        self.cliques: list[list[int]]
        self.sorted_vertices: list[int]
        self.heaviest_vertex_weight: int

        # load instance
        self.name = instance_file.split("/")[-1][:-4]
        self.nb_vertices, self.edges_list = read_col_files(instance_file)

        # miles instances have duplicate edges
        self.edges_list = list(set(self.edges_list))

        self.nb_edges = 0
        self.adjacency_matrix = [
            [False for _ in range(self.nb_vertices)] for _ in range(self.nb_vertices)
        ]
        self.neighborhood = [[] for _ in range(self.nb_vertices)]
        for vertex1, vertex2 in self.edges_list:
            if not self.adjacency_matrix[vertex1][vertex2]:
                self.nb_edges += 1
                self.adjacency_matrix[vertex1][vertex2] = True
                self.adjacency_matrix[vertex2][vertex1] = True
                self.neighborhood[vertex1].append(vertex2)
                self.neighborhood[vertex2].append(vertex1)

        # load weights
        self.weights = read_weights_file(weights_file, self.nb_vertices)

        # compute cliques
        self.cliques = [
            self.compute_clique_vertex(vertex) for vertex in range(self.nb_vertices)
        ]

        # sort vertices
        self.sorted_vertices = list(range(self.nb_vertices))
        self.sort_vertices()

        self.reduced_vertices = []
        self.nx_graph = nx.Graph()
        self.nx_graph.add_nodes_from(range(self.nb_vertices))
        for vertex1, vertex2 in self.edges_list:
            self.nx_graph.add_edge(vertex1, vertex2)

        self.color_map = ["white" for _ in range(self.nb_vertices)]
        self.color_map[0] = "#FF6E5D"
        self.color_map[1] = "#6BC4FF"
        self.color_map[2] = "#78FF91"
        self.color_map[3] = "#D59CFF"
        self.color_map[4] = "#FFF480"
        self.color_map[5] = "#FFBF91"
        self.number_map = list(range(self.nb_vertices))

        # self.pos = nx.shell_layout(self.nx_graph)
        # self.pos = nx.spring_layout(self.nx_graph)
        self.pos = [
            (2, 4),
            (1, 2.5),
            (2, 1),
            (4, 1),
            (5, 2.5),
            (4, 4),
            (1.5, 5.5),
            (2.5, 6.5),
            (3.5, 5.5),
            (4.5, 5.5),
            (1.5, 6.5),
            (0.5, 6),
            (4.5, 6.5),
        ]
        self.labels = {
            vertex: f"{to_vertex(vertex)}-{self.weights[vertex]}"
            for vertex in range(self.nb_vertices)
        }
        plt.figure(figsize=(20, 20))
        ax = plt.gca()
        ax.set_title("Original Graph", fontsize=40)
        edge_color = ["gray" for _ in self.nx_graph.edges()]
        for i, (vertex1, vertex2) in enumerate(self.nx_graph.edges()):
            for clique in self.cliques:
                if vertex1 in clique and vertex2 in clique:
                    edge_color[i] = "black"
        nx.draw(
            self.nx_graph,
            pos=self.pos,
            node_color=self.color_map,
            node_size=6000,
            font_size=28,
            labels=self.labels,
            with_labels=True,
            ax=ax,
            linewidths=2,
            edge_color=edge_color,
            edgecolors="black",
            width=5,
        )
        self.img_number = 0
        plt.savefig(
            f"img_reduction/{self.name}_{self.img_number:03d}.png", format="PNG"
        )
        # plt.show()
        plt.clf()
        plt.close()

    def remove_and_plot_R0(self, vertex, clique_r):
        self.color_map[vertex] = "#4DFF00"
        plt.figure(figsize=(20, 20))
        ax = plt.gca()
        ax.set_title(
            f"Remove {to_vertex(vertex)} with R0",
            fontsize=40,
        )
        edge_color = ["gray" for _ in self.nx_graph.edges()]
        for i, (vertex1, vertex2) in enumerate(self.nx_graph.edges()):
            for clique in self.cliques:
                if vertex1 in clique and vertex2 in clique:
                    edge_color[i] = "black"

        for i, (vertex1, vertex2) in enumerate(self.nx_graph.edges()):
            if vertex1 in clique_r and vertex2 in clique_r:
                edge_color[i] = "red"
        nx.draw(
            self.nx_graph,
            pos=self.pos,
            node_color=self.color_map,
            node_size=6000,
            font_size=28,
            labels=self.labels,
            with_labels=True,
            ax=ax,
            linewidths=2,
            edge_color=edge_color,
            edgecolors="black",
            width=5,
        )
        self.img_number += 1
        plt.savefig(
            f"img_reduction/{self.name}_{self.img_number:03d}.png",
            format="PNG",
        )
        plt.clf()
        # plt.show()
        plt.close()

        for vertex1, vertex2 in self.edges_list:
            if vertex1 == vertex and vertex2 not in self.reduced_vertices:
                self.nx_graph.remove_edge(vertex1, vertex2)
            if vertex2 == vertex and vertex1 not in self.reduced_vertices:
                self.nx_graph.remove_edge(vertex1, vertex2)

        plt.figure(figsize=(20, 20))
        ax = plt.gca()
        ax.set_title(
            f"Remove {to_vertex(vertex)} with R0",
            fontsize=40,
        )
        edge_color = ["gray" for _ in self.nx_graph.edges()]
        for i, (vertex1, vertex2) in enumerate(self.nx_graph.edges()):
            for clique in self.cliques:
                if vertex1 in clique and vertex2 in clique:
                    edge_color[i] = "black"

        for i, (vertex1, vertex2) in enumerate(self.nx_graph.edges()):
            if vertex1 in clique_r and vertex2 in clique_r:
                edge_color[i] = "red"
        nx.draw(
            self.nx_graph,
            pos=self.pos,
            node_color=self.color_map,
            node_size=6000,
            font_size=28,
            labels=self.labels,
            with_labels=True,
            ax=ax,
            linewidths=2,
            edge_color=edge_color,
            edgecolors="black",
            width=5,
        )
        self.img_number += 1
        plt.savefig(
            f"img_reduction/{self.name}_{self.img_number:03d}.png",
            format="PNG",
        )
        plt.clf()
        # plt.show()
        plt.close()

    def remove_and_plot_R1(self, vertex, clique_r, neighbors_clique_r):
        self.color_map[vertex] = "red"
        plt.figure(figsize=(20, 20))
        ax = plt.gca()
        ax.set_title(
            f"Remove {to_vertex(vertex)} with R1",
            fontsize=40,
        )
        edge_color = ["gray" for _ in self.nx_graph.edges()]
        for i, (vertex1, vertex2) in enumerate(self.nx_graph.edges()):
            for clique in self.cliques:
                if vertex1 in clique and vertex2 in clique:
                    edge_color[i] = "black"
        for i, (vertex1, vertex2) in enumerate(self.nx_graph.edges()):
            if vertex1 in clique_r and vertex2 in clique_r:
                edge_color[i] = "red"
            if vertex in (vertex1, vertex2) and (
                vertex1 in neighbors_clique_r or vertex2 in neighbors_clique_r
            ):
                edge_color[i] = "orange"
        nx.draw(
            self.nx_graph,
            pos=self.pos,
            node_color=self.color_map,
            node_size=6000,
            font_size=28,
            labels=self.labels,
            with_labels=True,
            ax=ax,
            linewidths=2,
            edge_color=edge_color,
            edgecolors="black",
            width=5,
        )
        self.img_number += 1
        plt.savefig(
            f"img_reduction/{self.name}_{self.img_number:03d}.png",
            format="PNG",
        )
        plt.clf()
        # plt.show()
        plt.close()

        for vertex1, vertex2 in self.edges_list:
            if vertex1 == vertex and vertex2 not in self.reduced_vertices:
                self.nx_graph.remove_edge(vertex1, vertex2)
            if vertex2 == vertex and vertex1 not in self.reduced_vertices:
                self.nx_graph.remove_edge(vertex1, vertex2)

        plt.figure(figsize=(20, 20))
        ax = plt.gca()
        ax.set_title(
            f"Remove {to_vertex(vertex)} with R1",
            fontsize=40,
        )
        edge_color = ["gray" for _ in self.nx_graph.edges()]
        for i, (vertex1, vertex2) in enumerate(self.nx_graph.edges()):
            for clique in self.cliques:
                if vertex1 in clique and vertex2 in clique:
                    edge_color[i] = "black"
        for i, (vertex1, vertex2) in enumerate(self.nx_graph.edges()):
            if vertex1 in clique_r and vertex2 in clique_r:
                edge_color[i] = "red"
        nx.draw(
            self.nx_graph,
            pos=self.pos,
            node_color=self.color_map,
            node_size=6000,
            font_size=28,
            labels=self.labels,
            with_labels=True,
            ax=ax,
            linewidths=2,
            edge_color=edge_color,
            edgecolors="black",
            width=5,
        )
        self.img_number += 1
        plt.savefig(
            f"img_reduction/{self.name}_{self.img_number:03d}.png",
            format="PNG",
        )
        plt.clf()
        # plt.show()
        plt.close()

    def remove_and_plot_R2(self, vertex, inter):
        self.color_map[vertex] = "yellow"

        plt.figure(figsize=(20, 20))
        ax = plt.gca()
        ax.set_title(
            f"Remove {to_vertex(vertex)} with R2 (dominated by {to_vertex(inter)})",
            fontsize=40,
        )
        vertices_inter = set(self.neighborhood[vertex]).intersection(
            self.neighborhood[inter]
        )
        edge_color = ["gray" for _ in self.nx_graph.edges()]
        for i, (vertex1, vertex2) in enumerate(self.nx_graph.edges()):
            for clique in self.cliques:
                if vertex1 in clique and vertex2 in clique and vertex not in clique:
                    edge_color[i] = "black"
            if vertex in (vertex1, vertex2):
                edge_color[i] = "orange"
            if inter in (vertex1, vertex2) and (
                vertex1 in vertices_inter or vertex2 in vertices_inter
            ):
                edge_color[i] = "red"
        nx.draw(
            self.nx_graph,
            pos=self.pos,
            node_color=self.color_map,
            node_size=6000,
            font_size=28,
            labels=self.labels,
            with_labels=True,
            ax=ax,
            linewidths=2,
            edge_color=edge_color,
            edgecolors="black",
            width=5,
        )
        self.img_number += 1
        plt.savefig(
            f"img_reduction/{self.name}_{self.img_number:03d}.png",
            format="PNG",
        )
        plt.clf()
        # plt.show()
        plt.close()

        for vertex1, vertex2 in self.edges_list:
            if vertex1 == vertex and vertex2 not in self.reduced_vertices:
                self.nx_graph.remove_edge(vertex1, vertex2)
            if vertex2 == vertex and vertex1 not in self.reduced_vertices:
                self.nx_graph.remove_edge(vertex1, vertex2)

        plt.figure(figsize=(20, 20))
        ax = plt.gca()
        ax.set_title(
            f"Remove {to_vertex(vertex)} with R2 (dominated by {to_vertex(inter)})",
            fontsize=40,
        )
        edge_color = ["gray" for _ in self.nx_graph.edges()]
        for i, (vertex1, vertex2) in enumerate(self.nx_graph.edges()):
            for clique in self.cliques:
                if vertex1 in clique and vertex2 in clique and vertex not in clique:
                    edge_color[i] = "black"
        nx.draw(
            self.nx_graph,
            pos=self.pos,
            node_color=self.color_map,
            node_size=6000,
            font_size=28,
            labels=self.labels,
            with_labels=True,
            ax=ax,
            linewidths=2,
            edge_color=edge_color,
            edgecolors="black",
            width=5,
        )
        self.img_number += 1
        plt.savefig(
            f"img_reduction/{self.name}_{self.img_number:03d}.png",
            format="PNG",
        )
        plt.clf()
        # plt.show()
        plt.close()

    def final_plot(self):
        plt.figure(figsize=(20, 20))
        ax = plt.gca()
        ax.set_title("Reduced Graph", fontsize=40)
        edge_color = ["gray" for _ in self.nx_graph.edges()]
        for i, (vertex1, vertex2) in enumerate(self.nx_graph.edges()):
            for clique in self.cliques:
                if vertex1 in clique and vertex2 in clique:
                    edge_color[i] = "black"
        nx.draw(
            self.nx_graph,
            pos=self.pos,
            node_color=self.color_map,
            node_size=6000,
            font_size=28,
            labels=self.labels,
            with_labels=True,
            ax=ax,
            linewidths=2,
            edge_color=edge_color,
            edgecolors="black",
            width=5,
        )
        self.img_number += 1
        plt.savefig(
            f"img_reduction/{self.name}_{self.img_number:03d}.png", format="PNG"
        )
        # plt.show()
        plt.clf()
        plt.close()

    def compute_clique_vertex(self, vertex) -> list[int]:
        """compute a clique for the vertex"""
        current_clique = [vertex]
        candidates = set(self.neighborhood[vertex])
        while candidates:
            # // choose next vertex than maximize
            # // b(v) = w(v) + (w(N(v) inter candidate) )/2
            best_vertex: int = -1
            best_benefit: float = -1
            for neighbor in candidates:
                commun_neighbors = candidates.intersection(self.neighborhood[neighbor])
                potential_weight = sum(self.weights[n] for n in commun_neighbors)
                benefit: float = self.weights[neighbor] + (potential_weight / 2)
                if benefit > best_benefit:
                    best_benefit = benefit
                    best_vertex = neighbor
            current_clique.append(best_vertex)
            candidates.remove(best_vertex)
            candidates = candidates.intersection(self.neighborhood[best_vertex])

        current_clique.sort(
            key=lambda v: (self.weights[v], len(self.neighborhood[v])), reverse=True
        )
        if len(current_clique) <= 2:
            return []
        return current_clique

    def sort_vertices(self) -> None:
        """sort the vertices and update heaviest vertex"""
        self.sorted_vertices.sort(
            key=lambda v: (
                self.weights[v],
                len(self.neighborhood[v]),
                sum(self.weights[n] for n in self.neighborhood[v]),
            ),
            reverse=True,
        )

    def delete_vertex(self, vertex: int) -> None:
        """Remove the vertex and add it to the list of reduced vertex"""
        recompute_clique = list(self.neighborhood[vertex])
        for neighbor in self.neighborhood[vertex]:
            self.neighborhood[neighbor].remove(vertex)
        self.neighborhood[vertex] = []
        self.reduced_vertices.append(vertex)
        self.cliques[vertex] = []
        for neighbor in recompute_clique:
            if vertex in self.cliques[neighbor]:
                self.cliques[neighbor] = self.compute_clique_vertex(neighbor)


def reduction_commun_neighbors(graph: Graph, vertex: int):
    """vertex a can be deleted if exist one vertex b neighbor to each neighbor of a with
    weight(b) >= weight(a)"""
    if not graph.neighborhood[vertex]:
        return False
    # List all neighbors of all neighbors
    neighbors = [set(graph.neighborhood[n]) for n in graph.neighborhood[vertex]]
    # List commun neighbors
    inter = neighbors[0].intersection(*neighbors)
    # Remove concerned vertex from the commun neighbors
    inter.remove(vertex)
    # For each commun neighbors sorted by weight and degree
    for n_vertex in inter:
        if graph.weights[n_vertex] >= graph.weights[vertex]:
            graph.remove_and_plot_R2(vertex, n_vertex)
            return True
    return False


def reduction_cliques(graph: Graph, vertex: int, use_neighbors_clique: bool) -> bool:
    """
    vertex can be deleted if some of its neighbors are
    in a clique and lower its degree according to the clique
    """
    vertex_weight: int = graph.weights[vertex]
    neighbors = set(graph.neighborhood[vertex])

    for clique in graph.cliques:
        if vertex in clique:
            continue
        d: int = len(graph.neighborhood[vertex]) + 1

        if use_neighbors_clique:
            for i, c_vertex in enumerate(reversed(clique)):
                if c_vertex in neighbors and len(clique) - i >= d:
                    d -= 1

        if len(clique) < d:
            continue
        if vertex_weight <= graph.weights[clique[d - 1]]:
            if d == len(graph.neighborhood[vertex]) + 1:
                graph.remove_and_plot_R0(vertex, clique[0:d])
            else:
                graph.remove_and_plot_R1(
                    vertex, clique, set(clique).intersection(neighbors)
                )
            return True

    return False


def reduce_instance(
    instance_name: str,
    problem: str,
    use_commun_neighbors: bool,
    use_neighbors_clique: bool,
    iterate: bool,
    reduced_directory: str,
):
    """reduce vertices for the graph"""
    # load graph and cliques
    start = time.time()
    instance_file: str = f"original_graphs/{instance_name}.col"
    weights_file: str = "" if problem == "gcp" else instance_file + ".w"
    if not os.path.exists(weights_file) and problem == "wvcp":
        raise Exception(f"No weights file (.col.w) for {instance_name}")

    graph: Graph = Graph(instance_file, weights_file)
    time_graph_loading = round(time.time() - start, 2)

    # reduction
    start = time.time()
    did_reduction = True
    nb_turns = 0
    nb_r1 = 0
    nb_r2 = 0
    while did_reduction:
        nb_turns += 1
        did_reduction = False
        graph.sort_vertices()
        for vertex in graph.sorted_vertices:
            if vertex in graph.reduced_vertices:
                continue
            # rule 1
            if use_commun_neighbors and reduction_commun_neighbors(graph, vertex):
                nb_r1 += 1
                did_reduction = True
                graph.delete_vertex(vertex)
            # rule 2
            elif reduction_cliques(graph, vertex, use_neighbors_clique):
                nb_r2 += 1
                did_reduction = True
                graph.delete_vertex(vertex)
        if not iterate:
            break
    time_reduction = round(time.time() - start, 2)

    # convert graph
    start = time.time()
    time_conversion = round(time.time() - start, 2)
    graph.final_plot()
    print(
        instance_name,
        graph.nb_vertices,
        graph.nb_edges,
        time_graph_loading,
        time_reduction,
        time_conversion,
        nb_turns,
        nb_r1,
        nb_r2,
        nb_r1 + nb_r2,
        sep=",",
    )


def compute_reduction_for_problem(
    problem: str,
    use_commun_neighbors: bool,
    use_neighbors_clique: bool,
    iterate: bool,
    reduced_directory: str,
):
    """find all instances for a problem"""
    print(
        "instance",
        "|V|",
        "|E|",
        "C time",
        "R time",
        "Conv time",
        "|iter|",
        "|R1|",
        "|R2|",
        "|R|",
        sep=",",
    )
    instances_names: list[str] = []
    with open(f"instance_list_{problem}.txt", "r", encoding="utf8") as instances_file:
        instances_names = instances_file.read().splitlines()
    instances_names = [
        "0_test",
        # "p31",
        # "inithx.i.1",
        # "miles250",
        # "p29",
        # "GEOM20a",
        # "inithx.i.2",
        # "GEOM30",
        # "GEOM20",
        # "miles500",
        # "inithx.i.3",
        # "zeroin.i.1",
        # "GEOM40a",
        # "zeroin.i.3",
        # "zeroin.i.2",
        # "GEOM40",
        # "GEOM50",
        # "DSJR500.1",
        # "GEOM80",
        # "mulsol.i.5",
        # "GEOM50a",
        # "GEOM20b",
        # "GEOM60",
        # "GEOM30a",
        # "p32",
        # "GEOM40b",
        # "miles1000",
        # "miles1500",
        # "GEOM60a",
        # "GEOM100",
        # "GEOM70a",
        # "GEOM50b",
        # "GEOM90b",
        # "le450_25b",
        # "le450_25a",
        # "GEOM70b",
        # "GEOM110",
        # "GEOM90",
        # "GEOM60b",
        # "GEOM110b",
        # "GEOM120",
        # "R50_1g",
        # "R50_1gb",
        # "p28",
        # "GEOM30b",
        # "GEOM80b",
        # "GEOM100b",
        # "GEOM70",
        # "GEOM80a",
        # "p26",
        # "wap02a",
        # "wap06a",
        # "wap05a",
        # "wap01a",
        # "p24",
        # "GEOM120b",
        # "p25",
        # "p36",
        # "le450_15a",
        # "wap04a",
        # "wap07a",
        # "wap08a",
        # "GEOM90a",
        # "le450_15b",
        # "p33",
        # "p35",
        # "wap03a",
        # "p21",
        # "p40",
        # "p38",
        # "le450_25c",
    ]
    for instance_name in instances_names:
        reduce_instance(
            instance_name,
            problem,
            use_commun_neighbors,
            use_neighbors_clique,
            iterate,
            reduced_directory,
        )


def main():
    """main function"""
    # base algorithm (RedLS article)
    # compute_reduction_for_problem(
    #     problem="wvcp",
    #     use_commun_neighbors=False,
    #     use_neighbors_clique=False,
    #     iterate=False,
    #     reduced_directory="reduced_1",
    # )

    # add commun neighbors rule
    # compute_reduction_for_problem(
    #     problem="wvcp",
    #     use_commun_neighbors=False,
    #     use_neighbors_clique=True,
    #     iterate=False,
    #     reduced_directory="reduced_2",
    # )

    # add neighbors to cliques rules
    # compute_reduction_for_problem(
    #     problem="wvcp",
    #     use_commun_neighbors=True,
    #     use_neighbors_clique=True,
    #     iterate=False,
    #     reduced_directory="reduced_3",
    # )

    # add iterations
    # compute_reduction_for_problem(
    #     problem="wvcp",
    #     use_commun_neighbors=True,
    #     use_neighbors_clique=True,
    #     iterate=True,
    #     reduced_directory="reduced_wvcp",
    # )

    # for gcp
    # compute_reduction_for_problem(
    #     problem="gcp",
    #     use_commun_neighbors=True,
    #     use_neighbors_clique=True,
    #     iterate=True,
    #     reduced_directory="reduced_gcp",
    # )

    compute_reduction_for_problem(
        problem="wvcp",
        use_commun_neighbors=True,
        use_neighbors_clique=True,
        iterate=True,
        reduced_directory="reduced_wvcp",
    )


if __name__ == "__main__":
    main()
