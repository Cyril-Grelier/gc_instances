"""
Reduce instances

"""
from __future__ import annotations
from dataclasses import dataclass, field
import os

import time


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


def write_to_file(table: str, file: str) -> None:
    """Write the table in the file (overwrite it if exist)"""
    with open(file, "w", encoding="utf8") as file_:
        file_.write(table)


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

    def convert_to_nodes(self, output_file_base: str, problem: str):
        """
        Convert graph to node representation before simplifying it and save it in different format
        """
        # Create the nodes
        nodes: list[Node] = [
            Node(
                old_number=vertex,
                new_number=-1,
                weight=self.weights[vertex],
                neighbors_int=self.neighborhood[vertex][:],
                neighbors_nodes=[],
                reduced=vertex in self.reduced_vertices,
            )
            for vertex in range(self.nb_vertices)
        ]
        # Add the neighbors to the nodes
        for node in nodes:
            node.neighbors_nodes = [nodes[neighbor] for neighbor in node.neighbors_int]
        # Sort the nodes by weights and degree
        nodes_not_reduced = sorted(
            [node for node in nodes if not node.reduced],
            key=lambda n: (
                n.weight,
                len(n.neighbors_int),
                sum(ne.weight for ne in n.neighbors_nodes),
            ),
            reverse=True,
        )
        # Gives the new numbers to the nodes
        for i, node in enumerate(nodes_not_reduced):
            node.new_number = i
            assert all(not n.reduced for n in node.neighbors_nodes)
        # Count the edges for DIMACS format
        nb_edges = 0
        for node in nodes_not_reduced:
            node.neighbors_nodes.sort(key=lambda n: n.new_number)
            nb_edges += len(
                [n for n in node.neighbors_nodes if n.new_number < node.new_number]
            )
        # Prepare text for conversion file
        txt_conv = (
            f"c conversion from graph {self.name} to reduce version\n"
            "c lines starting with c : comments\n"
            "c lines starting with d : the first number is the number "
            "of the vertex in original graph, the second in the reduced graph\n"
            "c lines starting with r : the vertices can be colored with the first "
            "available color without increasing the score\n"
        )
        for vertex in reversed(self.reduced_vertices):
            txt_conv += f"r {vertex}\n"

        for node in nodes:
            if not node.reduced:
                txt_conv += f"d {node.old_number} {node.new_number}\n"
        write_to_file(txt_conv, f"{output_file_base}.conv")

        # Prepare different format
        write_to_file(
            "".join(
                [
                    f"{node.new_number} {n_node.new_number}\n"
                    for node in nodes_not_reduced
                    for n_node in node.neighbors_nodes
                    if node.new_number < n_node.new_number
                ]
            ),
            f"{output_file_base}.edgelist",
        )

        txt_col_file = (
            f"c Reduced graph for {self.name} generated by Cyril Grelier\n"
            + f"p edge {len(nodes_not_reduced)} {nb_edges}\n"
        )
        nb_edges_v = 0
        for node in nodes_not_reduced:
            for n_node in node.neighbors_nodes:
                if node.new_number < n_node.new_number:
                    line = f"e {node.new_number +1 } {n_node.new_number + 1}\n"
                    txt_col_file += line
                    nb_edges_v += 1
        assert nb_edges == nb_edges_v

        write_to_file(txt_col_file, f"{output_file_base}.col")

        if problem == "wvcp":
            # Weights file
            write_to_file(
                "".join([f"{node.weight}\n" for node in nodes_not_reduced]),
                f"{output_file_base}.col.w",
            )


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
    graph.convert_to_nodes(f"{reduced_directory}/{instance_name}", problem)
    time_conversion = round(time.time() - start, 2)

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
    compute_reduction_for_problem(
        problem="gcp",
        use_commun_neighbors=True,
        use_neighbors_clique=True,
        iterate=True,
        reduced_directory="reduced_gcp",
    )


if __name__ == "__main__":
    main()
