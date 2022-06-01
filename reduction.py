"""
Code to reduce graphs
"""
from __future__ import annotations
from dataclasses import dataclass, field
import os
import multiprocessing
from glob import glob

import igraph


def main():
    # to reduce all instances (listed in instance_list.txt)
    # reduction_all(timeout=20, problem="wvcp")
    reduction_all(timeout=20, problem="gcp")


@dataclass
class Node:
    """Representation of a Node for graph"""

    old_number: int = -1
    new_number: int = -1
    weight: int = field(default_factory=list)
    neighbors_int: list[int] = field(default_factory=list)
    neighbors_nodes: list[Node] = field(default_factory=list)
    reduced: bool = False

    def __repr__(self):
        return (
            f"old_number={self.old_number} new_number={self.new_number} "
            + f"weight={self.weight} degree={len(self.neighbors_int)}"
        )


@dataclass
class Graph:
    """Representation of a graph"""

    name: str = ""
    nb_vertices: int = 0
    edges_list: list[tuple[int, int]] = field(default_factory=list)
    adjacency_matrix: list[list[bool]] = field(default_factory=list)
    neighborhood: list[list[int]] = field(default_factory=list)
    weights: list[int] = field(default_factory=list)
    reduced_vertices: list[int] = field(default_factory=list)
    second_reduction: dict[int, int] = field(default_factory=dict)
    cliques: list[list[int]] = field(default_factory=list)

    def delete_vertex(self, vertex: int) -> None:
        """Remove the vertex and add it to the list of reduced vertex

        :param vertex: the vertex to remove
        :type vertex: int
        """
        for neighbor in self.neighborhood[vertex]:
            self.neighborhood[neighbor].remove(vertex)
        self.neighborhood[vertex] = []
        self.reduced_vertices.append(vertex)

    def get_heavier_higher_degree(self) -> tuple[int, int]:
        """Get the weight and degree of the heavier vertex

        Returns:
            tuple[int, int]: max weight, max degree
        """
        max_weight = max(self.weights)
        max_vertex_degree = 0
        for vertex, neighbors in enumerate(self.neighborhood):
            if self.weights[vertex] == max_weight:
                degree = len(neighbors)
                if degree > max_vertex_degree:
                    max_vertex_degree = degree
        return max_weight, max_vertex_degree

    def reduction_1(self) -> int:
        """Apply the reduction based on cliques"""
        if not self.cliques:
            return 0

        # Size largest clique
        size_clique_max: int = max([len(clique) for clique in self.cliques])
        to_check = []
        to_delete = []
        for vertex in range(self.nb_vertices):
            vertex_degree: int = len(self.neighborhood[vertex])
            # if its degree is lower than the size of the largest clique
            # and its weight is lower than the weight of any
            # vertex of all cliques in the column of its degree
            if vertex_degree < size_clique_max and self.weights[vertex] < max(
                [
                    self.weights[clique[vertex_degree]]
                    for clique in self.cliques
                    if len(clique) > vertex_degree
                ]
            ):
                # the vertex can be deleted
                to_delete.append(vertex)
            elif not self.neighborhood[vertex]:
                to_check.append(vertex)
        # delete vertices
        for vertex in to_delete:
            self.delete_vertex(vertex)
        max_vertex_weight, max_vertex_degree = self.get_heavier_higher_degree()
        # look for vertices of maximum weight but no neighbors
        to_delete_post = []
        for vertex in to_check:
            if self.weights[vertex] <= max_vertex_weight and max_vertex_degree > 0:
                to_delete_post.append(vertex)
        for vertex in to_delete_post:
            self.delete_vertex(vertex)
        self.reduced_vertices.sort(
            key=lambda v: (
                self.weights[v],
                len(self.neighborhood[v]),
            ),
            reverse=True,
        )
        return len(to_delete) + len(to_delete_post)

    def reduction_2(self) -> int:
        """Apply the reduction based on neighborhood"""
        list_vertices = sorted(
            [v for v in range(self.nb_vertices) if v not in self.reduced_vertices],
            key=lambda v: (self.weights[v], len(self.neighborhood[v])),
        )
        to_delete = []
        # For each free vertex
        for vertex in list_vertices:
            # If the vertex isn't used to reduce an other one
            # (may be useless as we delete with the heavier one)
            if vertex in self.second_reduction.values():
                continue
            # List all neighbors of all neighbors
            neighbors = [set(self.neighborhood[n]) for n in self.neighborhood[vertex]]
            if neighbors:
                # List commun neighbors
                inter = neighbors[0].intersection(*neighbors)
                # Remove conserned vertex from the commun neighbors
                inter.remove(vertex)
                assert not any(n_vertex in self.reduced_vertices for n_vertex in inter)
                # For each commun neighbors sorted by weight and degree
                for n_vertex in sorted(
                    inter,
                    key=lambda v: (self.weights[v], len(self.neighborhood[v])),
                    reverse=True,
                ):
                    # if the neighbor is heavier than the vertex
                    if (
                        self.weights[n_vertex] > self.weights[vertex]
                        or (
                            self.weights[n_vertex] == self.weights[vertex]
                            and len(self.neighborhood[vertex])
                            < len(self.neighborhood[n_vertex])
                        )
                        or (
                            self.weights[n_vertex] == self.weights[vertex]
                            and len(self.neighborhood[vertex])
                            == len(self.neighborhood[n_vertex])
                            and n_vertex < vertex
                        )
                    ):
                        # the vertex can be deleted as it can take the color
                        # of the neighbor without increasing the score
                        self.second_reduction[vertex] = n_vertex
                        to_delete.append(vertex)
                        break
        # delete vertices
        for vertex in to_delete:
            self.delete_vertex(vertex)
        return len(to_delete)

    def convert_to_nodes(self, output_file_base: str, only_conv_ed_w: bool = True):
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
            for neighbor in node.neighbors_int:
                if nodes[neighbor].old_number != neighbor:
                    print("error")
            node.neighbors_nodes = [nodes[neighbor] for neighbor in node.neighbors_int]
        # Sort the nodes by weights and degree
        nodes_not_reduced = sorted(
            [node for node in nodes if not node.reduced],
            key=lambda n: (n.weight, len(n.neighbors_int)),
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
        if only_conv_ed_w:
            # Prepare text for conversion file
            txt_conv = (
                f"c conversion from graph {self.name} to reduce version\n"
                + "c lines starting with c : comments\n"
                + "c lines starting with d : the first number is the number "
                + "of the vertex in original graph, the second in the reduced graph\n"
                + "c lines starting with g : the vertex can be colored with an existing "
                + "color without increasing the score\n"
                + "c lines starting with s : the first vertex can be colored with the "
                + "color of the second vertex (numbers from original graph)\n"
            )
            for node in nodes:
                if node.reduced:
                    if node.old_number in self.second_reduction:
                        txt_conv += (
                            f"s {node.old_number} "
                            + f"{self.second_reduction[node.old_number]}\n"
                        )
                    else:
                        txt_conv += f"g {node.old_number}\n"
                else:
                    txt_conv += f"d {node.old_number} {node.new_number}\n"
            write_to_file(txt_conv, f"{output_file_base}.conv")

            # Prepare text for instance file
            txt_col_file = (
                f"c Reduced graph for {self.name} generated by Cyril Grelier\n"
                f"p edge {len(nodes_not_reduced)} {nb_edges}\n"
            )

            nb_edges_v = 0
            for node in nodes_not_reduced:
                for n_node in node.neighbors_nodes:
                    if node.new_number < n_node.new_number:
                        txt_col_file += (
                            f"e {node.new_number +1 } {n_node.new_number + 1}\n"
                        )
                        nb_edges_v += 1
            assert nb_edges == nb_edges_v

            write_to_file(txt_col_file, f"{output_file_base}.col")

            # Weights file
            write_to_file(
                "".join([f"{node.weight}\n" for node in nodes_not_reduced]),
                f"{output_file_base}.col.w",
            )

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

            return
        # Prepare different format
        txt_col_file = (
            f"c Reduced graph for {self.name} generated by Cyril Grelier\n"
            + f"p edge {len(nodes_not_reduced)} {nb_edges}\n"
        )
        txt_w_col_file = (
            f"c Reduced graph for {self.name} generated by Cyril Grelier\n"
            + f"p edge {len(nodes_not_reduced)} {nb_edges}\n"
        ) + "".join(
            [f"v {node.new_number + 1} {node.weight}\n" for node in nodes_not_reduced]
        )
        nb_edges_v = 0
        for node in nodes_not_reduced:
            for n_node in node.neighbors_nodes:
                if node.new_number < n_node.new_number:
                    line = f"e {node.new_number +1 } {n_node.new_number + 1}\n"
                    txt_col_file += line
                    txt_w_col_file += line
                    nb_edges_v += 1
        assert nb_edges == nb_edges_v

        write_to_file(txt_col_file, f"{output_file_base}.col")

        if any(node.weight != 1 for node in nodes_not_reduced):
            write_to_file(txt_w_col_file, f"{output_file_base}.wcol")
            write_to_file(
                "".join([f"{node.weight}\n" for node in nodes_not_reduced]),
                f"{output_file_base}.col.w",
            )

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


def load_graph(instance_file: str, weights_file: str, cliques_file: str) -> Graph:
    """Load graph file

    Args:
        instance_file (str): file containing the instance (edge list file)
        instance_weights_file (str): file containing the weights of the instance (col.w file)
        cliques_file (str): file containing the cliques of the instance (cliques file)

    Returns:
        Graph: graph from the file
    """
    nb_vertices = 0
    edges_list: list[tuple[int, int]] = []
    nb_edges: int = 0
    name: str = ""
    if instance_file.endswith("edgelist"):
        name = instance_file.split("/")[-1][:-9]
        # be careful with edgelist files
        # the number of vertices can be wrong if any of
        # the last vertices of the graph as no neighbors
        # (compared to the number of weights)
        with open(instance_file, "r", encoding="utf8") as file:
            for line in file.readlines():
                vertex1, vertex2 = sorted(list(map(int, line.split())))
                edges_list.append((vertex1, vertex2))
                nb_edges += 1
                if vertex2 > nb_vertices:
                    nb_vertices = vertex2
            nb_vertices += 1
    elif instance_file.endswith("col"):
        name = instance_file.split("/")[-1][:-4]
        with open(instance_file, "r", encoding="utf8") as file:
            for line_ in file:
                line = line_.strip()
                if line.startswith("c"):
                    continue
                elif line.startswith("p"):
                    try:
                        _, _, nb_vertices_str, nb_edges_str = line.split()
                    except ValueError as e:
                        print("p line does not conform but continue anyway")
                        _, _, nb_vertices_str, nb_edges_str, _ = line.split()
                    nb_vertices = int(nb_vertices_str)
                    nb_edges = int(nb_edges_str)
                elif line.startswith("e"):
                    _, vertex1_str, vertex2_str = line.split()
                    vertex1_ = int(vertex1_str) - 1
                    vertex2_ = int(vertex2_str) - 1
                    vertex1 = min(vertex1_, vertex2_)
                    vertex2 = max(vertex1_, vertex2_)
                    edges_list.append((vertex1, vertex2))
    else:
        print("error : not supported format for graph")
    weights = []
    if weights_file == "":
        weights = [1] * nb_vertices
    else:
        with open(weights_file, "r", encoding="utf8") as file:
            weights = list(map(int, file.readlines()))
    if len(weights) != nb_vertices:
        print("problem number of vertices")
    adjacency_matrix = [[False for _ in range(nb_vertices)] for _ in range(nb_vertices)]
    neighborhood: list[list[int]] = [[] for _ in range(nb_vertices)]
    for vertex1, vertex2 in edges_list:
        if not adjacency_matrix[vertex1][vertex2]:
            adjacency_matrix[vertex1][vertex2] = True
            adjacency_matrix[vertex2][vertex1] = True
            neighborhood[vertex1].append(vertex2)
            neighborhood[vertex2].append(vertex1)
    cliques: list[list[int]] = []
    # load cliques
    if cliques_file != "":
        with open(cliques_file, "r", encoding="utf8") as file:
            for line in file.readlines():
                # sort the vertices in the clique per weight
                cliques.append(
                    sorted(
                        list(map(int, line.split())),
                        key=lambda v: weights[v],
                        reverse=True,
                    )
                )
    # sort the cliques per total weight
    cliques.sort(key=lambda clique: sum(weights[v] for v in clique))
    return Graph(
        name=name,
        nb_vertices=nb_vertices,
        edges_list=edges_list,
        adjacency_matrix=adjacency_matrix,
        neighborhood=neighborhood,
        weights=weights,
        cliques=cliques,
    )


def write_to_file(table: str, file: str) -> None:
    """Write the table in the file (overwrite it if exist)

    :param table: The table to save
    :type table: list[str]
    :param file: file to write
    :type file: str
    """
    with open(file, "w", encoding="utf8") as file_:
        for line in table:
            file_.write(line)


def compute_cliques(instance_file: str, timeout: int, output_file: str):
    """Compute cliques with igraph

    Args:
        instance_file (str): Name of the edgelist file
        timeout (int): Max time to look for the cliques
        output_file (str): File name where the cliques will be listed
    """
    graph_g = igraph.Graph.Read_Edgelist(instance_file, directed=False)

    process = multiprocessing.Process(
        target=graph_g.maximal_cliques, args=(3, 0, output_file)
    )
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        print("Cliques partially loaded")


def reduction(instance_name: str, timeout: int, problem: str) -> tuple[int, int, int]:
    """Call the different phases of reduction until there is no more possible reduction

    Args:
        instance_name (str): Instance name
        timeout (int): Max time to compute the cliques
        problem (str): gcp or wvcp

    Returns:
        tuple[int,int,int]: number of vertices in original graph,
                            number of vertices deleted with first reduction,
                            number of vertices deleted with second reduction
    """
    print(instance_name)
    num_reduction: int = 0

    # load the original instance before sorting the vertices by weights
    instance_file = f"original_graphs/{instance_name}.col"
    weights_file = f"original_graphs/{instance_name}.col.w" if problem == "wvcp" else ""

    cliques_file = ""

    if problem not in ["gcp", "wvcp"]:
        print("error : choose problem between gcp and wvcp")
        exit(1)

    if not os.path.exists(weights_file) and problem == "wvcp":
        print("ignore instance as there is no weights file (.col.w)")
        return 0, 0, 0

    graph: Graph = load_graph(instance_file, weights_file, cliques_file)

    conversion_rep = f"conversion_{problem}"
    cliques_file = f"{conversion_rep}/{instance_name}.cliques"

    # keep track of the reduction
    nb_vertices = graph.nb_vertices
    total_reduction_1 = 0
    total_reduction_2 = 0
    reduction1 = 1
    reduction2 = 1
    while reduction1 or reduction2:
        # sort the nodes of the graph and save weights and edgelist files in {convertion_rep}/
        graph.convert_to_nodes(
            f"{conversion_rep}/{instance_name}_{num_reduction}", only_conv_ed_w=True
        )
        # compute the cliques with igraph
        if os.path.exists(cliques_file):
            os.remove(cliques_file)
        instance_file = f"{conversion_rep}/{instance_name}_{num_reduction}.col"
        instance_file_ = f"{conversion_rep}/{instance_name}_{num_reduction}.edgelist"
        weights_file = f"{conversion_rep}/{instance_name}_{num_reduction}.col.w"
        compute_cliques(
            instance_file_,
            timeout,
            cliques_file,
        )
        # Unix functions to analyse cliques :
        #   sed 's/[^ ]//g' tmp_cliques/C2000.9.cliques | awk '{print length }' | sort -u
        #   awk '{print NF,$0}' tmp_cliques/C2000.9.cliques | sort -nr | cut -d' ' -f 2-
        # load the graph with custom graph class and compute the reduction
        graph = load_graph(
            instance_file,
            weights_file,
            cliques_file,
        )

        reduction1 = graph.reduction_1()
        total_reduction_1 += reduction1
        reduction2 = graph.reduction_2()
        total_reduction_2 += reduction2
        num_reduction += 1
        print(f"Reduction {num_reduction} ({reduction1} + {reduction2})")

    if os.path.exists(cliques_file):
        os.remove(cliques_file)

    # save final reduced graph in wvcp_reduced_graphs/
    graph.convert_to_nodes(f"{problem}_reduced/{instance_name}", only_conv_ed_w=False)
    return nb_vertices, total_reduction_1, total_reduction_2


def reduction_all(timeout: int, problem: str):
    """reduce all instances with a edgelist file in wvcp_original"""
    with open(f"summary_reduction_{problem}.csv", "w", encoding="utf8") as output:
        output.write("instance,nb_vertices,first_reduction,second_reduction\n")
        for instance in sorted(glob("original_graphs/*.col")):
            inst = instance.split("/")[1][:-4]
            nb_vertices, nb_reduc1, nb_reduc2 = reduction(inst, timeout, problem)
            output.write(f"{inst},{nb_vertices},{nb_reduc1},{nb_reduc2}\n")


if __name__ == "__main__":
    main()
