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
import sys


def main():
    """check if the solution passed as input is valid"""
    if len(sys.argv) < 5:
        print("error input", file=sys.stderr)
        print(
            "expected : python check_solution instance_name problem score solution",
            file=sys.stderr,
        )
        print(f"got : {' '.join(sys.argv)}", file=sys.stderr)
        sys.exit(1)

    instance = sys.argv[1]
    problem = sys.argv[2]
    score = int(sys.argv[3])
    solution = list(map(int, sys.argv[4].split(":")))

    try:
        convert_solution(
            instance=instance,
            colors=solution,
            score=score,
            problem=problem,
        )
    except AssertionError as error:
        print(error)
        print(instance)
        print(problem)
        print(score)
        print(solution)
        sys.exit(1)


class Graph:
    """Representation of a graph"""

    def __init__(self, instance_file: str, weights_file: str) -> None:
        self.name: str = instance_file.split("/")[-1][:-4]
        self.nb_vertices: int = 0
        self.nb_edges: int = 0
        self.edges_list: list[tuple[int, int]] = []
        self.weights: list[int] = []

        with open(instance_file, "r", encoding="utf8") as file:
            for line_ in file:
                line = line_.strip()
                if line.startswith("c"):
                    continue
                if line.startswith("p"):
                    _, _, nb_vertices_str, nb_edges_str = line.split()
                    self.nb_vertices = int(nb_vertices_str)
                    self.nb_edges = int(nb_edges_str)
                elif line.startswith("e"):
                    _, vertex1_str, vertex2_str = line.split()
                    vertex1_ = int(vertex1_str) - 1
                    vertex2_ = int(vertex2_str) - 1
                    if vertex1_ == vertex2_:
                        continue
                    vertex1 = min(vertex1_, vertex2_)
                    vertex2 = max(vertex1_, vertex2_)
                    self.edges_list.append((vertex1, vertex2))

        if weights_file == "":
            self.weights = [1] * self.nb_vertices
        else:
            with open(weights_file, "r", encoding="utf8") as file:
                self.weights = list(map(int, file.readlines()))
        if len(self.weights) != self.nb_vertices:
            raise Exception(
                f"problem number of vertices in instance and weights {instance_file}"
            )

        self.adjacency_matrix: list[list[bool]] = [
            [False for _ in range(self.nb_vertices)] for _ in range(self.nb_vertices)
        ]
        self.neighborhood: list[list[int]] = [[] for _ in range(self.nb_vertices)]

        for vertex1, vertex2 in self.edges_list:
            if not self.adjacency_matrix[vertex1][vertex2]:
                self.adjacency_matrix[vertex1][vertex2] = True
                self.adjacency_matrix[vertex2][vertex1] = True
                self.neighborhood[vertex1].append(vertex2)
                self.neighborhood[vertex2].append(vertex1)


def load_conversion(file_name: str) -> tuple[dict[int, int], list[int]]:
    """Load conversion file to convert reduced solution to original one
    return different numbers for vertices and list of reduced vertices"""
    with open(file_name, "r", encoding="utf8") as file:
        different_number: dict[int, int] = {}
        reduced: list[int] = []
        for line in file.readlines():
            if line[0] == "d":
                _, number1, number2 = line.split()
                different_number[int(number1)] = int(number2)
            elif line[0] == "r":
                _, number = line.split()
                reduced.append(int(number))
    return different_number, reduced


class Solution:
    """Representation of a solution"""

    def __init__(
        self,
        graph: Graph,
        colors: list[int],
        conversion: tuple[dict[int, int], list[int]],
    ):
        if conversion:
            different_number, reduced = conversion
        else:
            different_number = {v: v for v in range(graph.nb_vertices)}
            reduced = []
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
        # color the original graph with the colors of the reduce graph
        for old, new in different_number.items():
            self.add_vertex_to_color(old, colors[new])

        score_before = self.current_score
        # color the reduced vertices
        for vertex in reduced:
            possible_colors = [
                color
                for color in range(len(self.conflicts_colors))
                if self.conflicts_colors[color][vertex] == 0
                and self.get_delta_score(vertex, color) == 0
            ]
            possible_colors.sort(
                key=lambda c: max(self.colors_weights[c]), reverse=True
            )
            if not possible_colors:
                raise Exception(
                    "problem during placement of reduced vertices : creation of new color"
                )
            color = possible_colors[0]

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
            ), f"color {color} not in the range [0, {self.nb_colors}[ "
            assert (
                self.conflicts_colors[color][vertex] == 0
            ), f"conflict on the color for the vertex ({self.conflicts_colors[color][vertex]})"

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

    # check if the solution is valid with the reduced graph
    reduce_file = f"reduced_{problem}/{instance}.col"
    weights_file = f"reduced_wvcp/{instance}.col.w" if problem == "wvcp" else ""

    graph: Graph = Graph(reduce_file, weights_file)
    assert (
        len(colors) == graph.nb_vertices
    ), f"Problem number of colors : {len(colors)} vertices colored for a graph of {graph.nb_vertices} vertices"
    sol: Solution = Solution(graph, colors, None)
    sol.check_solution(score)

    # check if the solution is valid with the original graph
    original_file = f"original_graphs/{instance}.col"
    original_weight_file = (
        f"original_graphs/{instance}.col.w" if problem == "wvcp" else ""
    )
    graph: Graph = Graph(original_file, original_weight_file)
    conv_file = f"reduced_{problem}/{instance}.conv"
    sol = Solution(graph, colors, load_conversion(conv_file))
    sol.check_solution(score)


if __name__ == "__main__":
    main()
