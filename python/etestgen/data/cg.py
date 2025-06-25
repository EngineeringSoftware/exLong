import dataclasses
from typing import Dict, Set, Tuple, List

import seutil as su


class EdgeType:
    CALL = "C"
    OVERRIDE = "O"


@dataclasses.dataclass
class CallGraph:
    edges: Dict[int, Set[Tuple[int, str]]] = dataclasses.field(default_factory=dict)

    @classmethod
    def deserialize(cls, data) -> "CallGraph":
        cg = cls()
        cg.edges = {int(k): v for k, v in su.io.deserialize(data["edges"]).items()}
        return cg

    def get_edges_from(self, mid: int) -> Set[Tuple[int, str]]:
        """
        Get all edges starting from the node (specified by its key).

        Time complexity: O(log(V))
        """
        return self.edges.get(mid, set())

    def get_edges_to(self, mid: int) -> Set[Tuple[int, str]]:
        """
        Get all edges targeting to the node (specified by its key).

        Time complexity: O(E)
        """
        collected_edges: Set[Tuple[int, str]] = set()
        for from_key, to_labels in self.edges.items():
            for to_key, label in to_labels:
                if to_key == mid:
                    collected_edges.add((from_key, label))

        return collected_edges

    def get_overridden_edges_to(self, mid: int) -> Set[int]:
        """
        Get all 'O' edges targeting to the node (specified by its key).

        i.e. get all methods that override mid.
        """
        collected_edges: Set[int] = set()
        for from_key, to_labels in self.edges.items():
            for to_key, label in to_labels:
                if to_key == mid and label == EdgeType.OVERRIDE:
                    collected_edges.add(from_key)

        return collected_edges

    def get_call_paths(self, mid: int) -> List[List[int]]:
        """
        Get all possible call paths starting from the node (specified by its key).
        """
        stack = [(mid, [mid])]
        all_paths = []
        while stack:
            (vertex, path) = stack.pop()
            possible_called_nodes: Set[int] = set()
            directly_called_nodes: Set[int] = set()
            for n_tp in self.get_edges_from(vertex):
                if n_tp[1] == EdgeType.CALL:
                    directly_called_nodes.add(n_tp[0])
            possible_called_nodes = possible_called_nodes.union(directly_called_nodes)
            if vertex != mid:
                possible_called_nodes = possible_called_nodes.union(
                    self.get_overridden_edges_to(vertex)
                )
            # find the overriding nodes
            # visited_nodes = set()
            # override_methods = [
            #     n_tp
            #     for n_tp in self.get_overridden_edges_to(vertex)
            #     if n_tp not in visited_nodes
            # ]
            # while len(override_methods) > 0:
            #     om_id = override_methods.pop(0)
            #     assert om_id not in visited_nodes
            #     if om_id not in possible_called_nodes:
            #         possible_called_nodes.add(om_id)
            #         visited_nodes.add(om_id)
            #         override_methods.extend(
            #             [
            #                 n_tp
            #                 for n_tp in self.get_overridden_edges_to(om_id)
            #                 if n_tp not in visited_nodes
            #             ]
            #         )
            for nid in possible_called_nodes:
                if nid not in path:
                    if (
                        len(
                            [
                                e
                                for e in self.get_edges_from(nid)
                                if e[1] == EdgeType.CALL
                            ]
                        )
                        == 0
                    ):
                        # the leaf node w. no outgoing edges
                        all_paths.append(path + [nid])
                    else:
                        stack.append((nid, path + [nid]))
        if len(all_paths) == 0:
            all_paths.append([mid])
        return all_paths

    def add_edge(self, from_mid: int, to_mid: int, edge_type: EdgeType = EdgeType.CALL):
        self.edges.setdefault(from_mid, set()).add((to_mid, edge_type))
