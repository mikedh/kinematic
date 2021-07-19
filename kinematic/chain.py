"""
kinematic.py
--------------

PROTOTYPE for support for rigid body kinematics. Not sure if
supporting this is something that is feasible or desirable for trimesh,
this is just exploring what it might look like. If this improves, it
could feasibly live as `trimesh.kinematic`.

Challenge: create a data structure which can (mostly) hold and
cross-convert GLTF Skeletons, OpenRave XML Robots, and URDF robots.

Uses sympy to produce numpy-lambdas for forward kinematics, which once computed
are quite fast (for Python anyway) to execute.
"""
import trimesh

import sympy as sp
import numpy as np
import networkx as nx

from .link import Link
from .joint import Joint, RotaryJoint, LinearJoint


class KinematicChain(object):
    """
    A mechanism which consists of geometry (`Link` objects) connected
    by variable transforms (`Joint` objects).
    """

    def __init__(self,
                 joints,
                 links,
                 base_link='base'):
        """
        Create a kinematic chain.

        Parameters
        --------------
        joints : dict
          Joint name to `Joint` objects
        links : (m,) Link object
          Link name to `Link` objects
        base_link : str
          Name of base link
        """
        # save passed joints and links
        self.joints = joints
        self.links = links

        # which link is the first
        self.base_link = base_link

    @property
    def base_frame(self):
        # TODO : figure something out here
        return self.base_link

    @property
    def parameters(self):
        """
        What are the variables that define the state of the chain.

        Returns
        ---------
        parameters : (n,) sympy.Symbol
          Ordered parameters
        """
        return [i.parameter for i in self.joints.values()]

    @property
    def limits(self):
        limits = np.sort([j.limits for j in
                          self.joints.values()], axis=1)
        return limits

    def graph(self):
        """
        Get a directed graph where joints are edges between links.

        Returns
        ----------
        graph : networkx.DiGraph
          Graph containing connectivity information
        """
        graph = nx.DiGraph()
        for name, joint in self.joints.items():
            graph.add_edge(*joint.connects, joint=name)
        return graph

    def scene(self):
        """
        Get a scene containing the geometry for every link.

        Returns
        -----------
        scene : trimesh.Scene
          Scene with link geometry
        """
        geometry = {}
        for name, link in self.links.items():
            geometry.update(link.geometry)

        base_frame = self.base_frame
        graph = trimesh.scene.transforms.SceneGraph()
        graph.from_edgelist([(base_frame, geom_name,
                              {'geometry': geom_name})
                             for geom_name in geometry.keys()])
        graph.update(frame_from=graph.base_frame, frame_to=base_frame)

        scene = trimesh.Scene(geometry, graph=graph)
        return scene

    def show(self):
        """
        Open a pyglet window showing all geometry.
        """
        self.scene().show()

    def paths(self):
        """
        Find the route from the base body to every link.

        Returns
        ---------
        joint_paths : dict
          Keys are link names, values are a list of joint objects
        """
        base = self.base_link
        graph = self.graph()
        paths = {}
        for b in self.links.values():
            try:
                paths[b.name] = shortest(graph, base, b.name)
            except BaseException as E:
                print('exception:', E)

        joint_paths = {}
        for body, path in paths.items():
            joint_paths[body] = [graph.get_edge_data(a, b)['joint']
                                 for a, b in zip(path[:-1], path[1:])]
        return joint_paths

    def forward_kinematics(self):
        """
        Get the symbolic sympy forward kinematics.

        Returns
        -----------
        symbolic : dict
          Keyed by body to a sympy matrix
        """
        def product(L):
            if len(L) == 0:
                return sp.Matrix.eye(4)
            cum = L[0]
            for i in L[1:]:
                cum *= i
            return cum

        # routes to base link
        paths = self.paths()
        # symbolic matrices
        matrices = {name: j.matrix for name, j in self.joints.items()}

        #
        combined = {k: product([matrices[i] for i in path])
                    for k, path in paths.items()}

        return combined

    def forward_kinematics_lambda(self):
        """
        Get a numpy-lambda for evaluating forward kinematics relatively
        quickly.

        Returns
        -----------
        lambdas : dict
          Link name to function which takes float values
          corresponding to self.parameters.
        """
        # a symbolic equation for every link
        combined = self.forward_kinematics()
        return {k: sp.lambdify(self.parameters, c)
                for k, c in combined.items()}


def shortest(graph, a, b):
    """
    Try to find a shortest path between two nodes.

    Parameters
    -------------
    graph : networkx.DiGraph
      Graph with nodes
    a : str
      Source node
    b : str
      Destination node

    Returns
    ----------
    path : (n,) str
      Path between `a` and `b`
    """
    try:
        s = nx.shortest_path(graph, a, b)
        return s
    except BaseException:
        # try traversing the DiGraph backwards
        s = nx.shortest_path(graph, b, a)
        return s[::-1]
