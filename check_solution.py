"""
Use this script to check if a solution is good.
4 parameters : name of the instance, problem (gcp/wvcp), score, colors (separated with ':')

example (good, no output):
    python3 check_solution.py p06 wvcp 565 0:0:1:1:1:1:2:2:2:2:0:0:4:4:3:3

example (error, print the assertion and exit 1):
    python3 check_solution.py p06 wvcp 500 0:0:1:1:1:1:2:2:2:2:0:0:4:4:3:3

"""
from __future__ import annotations
import bisect
from glob import glob
import sys
from dataclasses import dataclass, field


def main():
    instance: str = ""
    problem: str = ""
    score: int = 0
    solution: list[int] = []
    try:
        instance = sys.argv[1]
        problem = sys.argv[2]
        score = int(sys.argv[3])
        solution = list(map(int, sys.argv[4].split(":")))
    except IndexError as e:
        print("error input, got : ", sys.argv)
        print(
            "example : python check_solution.py p06 wvcp 565 0:0:1:1:1:1:2:2:2:2:0:0:4:4:3:3"
        )
        exit(1)
    try:
        convert_solution(
            instance=instance,
            colors=solution,
            score=score,
            problem=problem,
        )
    except AssertionError as e:
        print(e)
        print(instance)
        print(problem)
        print(score)
        print(solution)
        exit(1)


@dataclass
class Graph:
    """Representation of a graph"""

    name: str = ""
    nb_vertices: int = 0
    edges_list: list[tuple[int, int]] = field(default_factory=list)
    adjacency_matrix: list[list[bool]] = field(default_factory=list)
    neighborhood: list[list[int]] = field(default_factory=list)
    weights: list[int] = field(default_factory=list)


def load_graph(instance_file: str, weights_file: str) -> Graph:
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
    return Graph(
        name=name,
        nb_vertices=nb_vertices,
        edges_list=edges_list,
        adjacency_matrix=adjacency_matrix,
        neighborhood=neighborhood,
        weights=weights,
    )


def load_conversion(
    file_name: str,
) -> tuple[dict[int, int], list[int], dict[int, int]]:
    """Load conversion file to convert reduced solution to original one

    :param instance_name: name of the instance (without the _r)
    :type instance_name: str
    :return: different_number, greedy and same_color structures to convert the solution
    :rtype: tuple[dict[int, int], list[int], dict[int, int]]
    """
    with open(file_name, "r", encoding="utf8") as file:
        different_number: dict[int, int] = {}
        greedy: list[int] = []
        same_color: dict[int, int] = {}
        for line in file.readlines():
            if line[0] == "d":
                _, number1, number2 = line.split()
                different_number[int(number1)] = int(number2)
            elif line[0] == "g":
                _, number = line.split()
                greedy.append(int(number))
            elif line[0] == "s":
                _, number1, number2 = line.split()
                same_color[int(number1)] = int(number2)
    return different_number, greedy, same_color


class Solution:
    """Representation of a solution"""

    def __init__(
        self,
        graph: Graph,
        colors: list[int],
        conversion: tuple[dict[int, int], list[int], dict[int, int]],
    ):
        if conversion:
            different_number, greedy, same_color = conversion
        else:
            different_number = {v: v for v in range(graph.nb_vertices)}
            greedy = []
            same_color = dict()
        self.graph: Graph = graph
        self.nb_colors = max(colors) + 1
        # List of vertices of each colors (nb_colors x nb_vertices_per_color)
        self.color_vertices: list[list[int]] = [[] for _ in range(self.nb_colors)]
        # Colors for each vertices (nb_vertices)
        self.colors: list[int] = [-1] * graph.nb_vertices
        # Conflicts of each vertices for each colors (nb_colors x nb_vertices)
        self.conflicts_colors: list[list[int]] = [
            [0] * self.graph.nb_vertices for _ in range(self.nb_colors)
        ]
        # List of weights in each colors (nb_colors x nb_vertices_per_color)
        self.colors_weights: list[list[int]] = [[] for _ in range(self.nb_colors)]
        # Current score
        self.current_score: int = 0
        for old, new in different_number.items():
            self.add_vertex_to_color(old, colors[new])
        for vertex1, vertex2 in same_color.items():
            self.add_vertex_to_color(vertex1, self.colors[vertex2])

        score_before = self.current_score
        greedy.sort(
            key=lambda vertex: (graph.weights[vertex], len(graph.neighborhood[vertex]))
        )
        for vertex in greedy:
            possible_colors = [
                color
                for color in range(len(self.conflicts_colors))
                if self.conflicts_colors[color][vertex] == 0
                and self.get_delta_score(vertex, color) == 0
            ]
            if not possible_colors:
                raise Exception(
                    "problem during placement of reduced vertices : creation of new color"
                )
            color = (
                possible_colors[0] if possible_colors else len(self.conflicts_colors)
            )

            self.add_vertex_to_color(vertex, color)
            if score_before < self.current_score:
                raise Exception(
                    "problem during placement of reduced vertices : score increase"
                )

    def add_vertex_to_color(self, vertex: int, color: int) -> None:
        """
        Add the vertex to its new color

        :param vertex:
        :param color:
        """
        assert self.colors[vertex] == -1, f"vertex {vertex} color already set"
        assert color != -1, "can't add vertex to no color"
        assert len(self.colors_weights) == len(self.conflicts_colors)
        assert (
            len(self.conflicts_colors) <= color
            or self.conflicts_colors[color][vertex] == 0
        ), f"conflicts on the color {color} for vertex {vertex}"
        self.current_score += self.get_delta_score(vertex, color)
        # increase the number of conflict with the neighbors
        for neighbor in self.graph.neighborhood[vertex]:
            self.conflicts_colors[color][neighbor] += 1

        # insert the vertex in the existing color
        bisect.insort(self.colors_weights[color], self.graph.weights[vertex])
        bisect.insort(self.color_vertices[color], vertex)
        # Set the color of the vertex
        self.colors[vertex] = color

    def get_delta_score(self, vertex: int, color: int) -> int:
        """
        Compute the difference on the score if the vertex is move to the color

        :param vertex: the vertex to move
        :type vertex: int
        :param color: the new color
        :type color: int
        :return: the difference on the score if the vertex is set to the color
        """
        vertex_weight: int = self.graph.weights[vertex]
        # if the new color is empty
        if (len(self.colors_weights) <= color) or (not self.colors_weights[color]):
            # the delta is the weight of the vertex
            return vertex_weight

        # if the vertex is heavier than the heaviest of the new color class
        if vertex_weight > self.colors_weights[color][-1]:
            # the delta is the difference between the vertex weight and the heavier vertex
            return vertex_weight - self.colors_weights[color][-1]
        return 0

    def check_solution(self, score_val: int) -> None:
        """
        Check if the current score is correct depending on colors list
        """
        score = 0
        max_colors_weights: list[int] = [0] * self.nb_colors
        for vertex in range(self.graph.nb_vertices):
            color: int = self.colors[vertex]
            if color == -1:
                continue

            assert (
                0 <= color < self.nb_colors
                and self.conflicts_colors[color][vertex] == 0
            ), (
                f"color {color} not in the range [0, {self.nb_colors}[ "
                f"or conflict on the color for the vertex ({self.conflicts_colors[color][vertex]})"
            )

            weight: int = self.graph.weights[vertex]

            if max_colors_weights[color] < weight:
                max_colors_weights[color] = weight

            for neighbor in self.graph.neighborhood[vertex]:
                assert (
                    color != self.colors[neighbor]
                ), f"{vertex} and {neighbor} (neighbors) share the same color ({color})"

        for col in range(self.nb_colors):
            assert len(self.color_vertices[col]) == len(
                self.colors_weights[col]
            ), f"problem in color {col}"
            if not self.color_vertices[col]:
                continue
            assert (
                max_colors_weights[col] == self.colors_weights[col][-1]
            ), f"error in max weight of color {col}"
            score += max_colors_weights[col]

        assert (
            score == self.current_score
        ), f"Problem score {score} vs {self.current_score}"
        assert score == score_val, f"Problem on given score {score} vs {score_val}"


def convert_solution(
    instance: str,
    colors: list[int],
    score: int,
    problem: str,
):
    """Convert a solution from reduced graph to original graph

    Args:
        instance (str): instance name
        colors (list[int]): colors of the vertices
        score (int): score estimated
        problem (str): type of problem (gcp or wvcp)
    """
    conv_files = sorted(
        glob(f"conversion_{problem}/{instance}_*.conv"),
        key=lambda f: int(f.rsplit("_", 1)[1].split(".")[0]),
        reverse=True,
    )
    edge_files = sorted(
        glob(f"conversion_{problem}/{instance}_*.col"),
        key=lambda f: int(f.rsplit("_", 1)[1].split(".")[0]),
        reverse=True,
    )[1::] + [f"original_graphs/{instance}.col"]
    weights_files = []
    if problem == "wvcp":
        weights_files = sorted(
            glob(f"conversion_{problem}/{instance}_*.col.w"),
            key=lambda f: int(f.rsplit("_", 1)[1].split(".")[0]),
            reverse=True,
        )[1::] + [f"original_graphs/{instance}.col.w"]
    else:
        weights_files = [""] * len(edge_files)
    graph = load_graph(
        f"{problem}_reduced/{instance}.col",
        f"{problem}_reduced/{instance}.col.w" if problem == "wvcp" else "",
    )
    sol = Solution(graph, colors, None)
    sol.check_solution(score)
    for conv_file, edge_file, weights_file in zip(
        conv_files, edge_files, weights_files
    ):
        graph: Graph = load_graph(edge_file, weights_file)
        conversion = load_conversion(conv_file)
        sol = Solution(graph, colors, conversion)
        sol.check_solution(score)
        colors = sol.colors[:]


if __name__ == "__main__":
    main()
