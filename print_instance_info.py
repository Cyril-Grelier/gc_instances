"""
Print instance informations

"""
from __future__ import annotations


def main():
    """main function"""
    problem = "gcp"
    print("instance", "reduced", "|V|", "|E|", "density", sep=",")
    instances_names: list[str] = []

    with open(f"instance_list_{problem}.txt", "r", encoding="utf8") as instances_file:
        instances_names = instances_file.read().splitlines()

    for instance_name in instances_names:
        instance_file: str = f"original_graphs/{instance_name}.col"
        weights_file: str = "" if problem == "gcp" else instance_file + ".w"
        graph: Graph = Graph(instance_file, weights_file)
        print(
            graph.name,
            "false",
            graph.nb_vertices,
            graph.nb_edges,
            graph.density,
            sep=",",
        )

        instance_file = f"reduced_{problem}/{instance_name}.col"
        weights_file = "" if problem == "gcp" else instance_file + ".w"
        graph: Graph = Graph(instance_file, weights_file)
        print(
            graph.name,
            "true",
            graph.nb_vertices,
            graph.nb_edges,
            graph.density,
            sep=",",
        )


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
        self.density: float

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

        # compute density
        self.density = round(
            (2 * self.nb_edges) / (self.nb_vertices * (self.nb_vertices - 1)), 2
        )


if __name__ == "__main__":
    main()
