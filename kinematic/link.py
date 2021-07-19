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

ABC = trimesh.util.ABC


class Link(object):
    def __init__(self, name, geometry):
        """
        `Link` objects store geometry.

        Parameters
        ------------
        name : str
          The name of the Link object
        geometry : dict
          Any geometry that this link contains
        """
        self.name = name
        self.geometry = geometry

    def show(self, **kwargs):
        trimesh.Scene(self.geometry).show(**kwargs)
