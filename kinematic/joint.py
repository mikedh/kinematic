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

import abc
import sympy as sp
import numpy as np

ABC = trimesh.util.ABC
# for debugging
from trimesh.exchange.threedxml import print_element as pp  # NOQA

try:
    import lxml.etree as etree
except BaseException as E:
    etree = trimesh.exceptions.ExceptionModule(E)


class Joint(ABC):
    """
    The base class for `Joint` objects, or connections
    between `Link` objects which contain geometry.
    """
    @abc.abstractmethod
    def matrix(self):
        """
        The symbolic homogenous transformation matrix between
        `self.connects[0]` and `self.connects[1]`.

        Returns
        -----------
        matrix : sympy.Matrix
          Transform with `self.parameter` as a variable
        """
        raise NotImplementedError('call a subclass!')

    @property
    def connects(self):
        """
        The name of the two links this joint is connecting.

        Returns
        -------------
        connects : (2,) list
          The name of two `Link` objects
        """
        return self._connects

    @connects.setter
    def connects(self, values):
        if values is None or len(values) != 2:
            raise ValueError('`connects` must be two link names!')
        self._connects = values

    @property
    def limits(self):
        """
        The
        """
        if hasattr(self, '_limits'):
            return self._limits
        return [-np.inf, np.inf]

    @limits.setter
    def limits(self, values):
        if values is not None:
            self._limits = values


class RotaryJoint(Joint):
    def __init__(self,
                 name,
                 axis,
                 connects,
                 initial=None,
                 limits=None,
                 anchor=None):
        """
        Create a rotary joint between two links.

        Parameters
        -------------
        name : str
          The name of this joint.
        axis : (3,) float
          The unit vector this joint revolves around.
        connects : (2,) str
          The name of the two `Link` objects this joint connects
        initial : None or (4, 4) float
          Initial transformation.
        limits : None or (2,) float
          The limits of this joint in radians.
        anchor : None or (3,) float
          The point in space anchoring this joint,
          also known as the origin of the axis line
        """
        # the unit vector axis
        self.axis = np.array(axis, dtype=np.float64).reshape(3)
        # the point around which to rotate
        if anchor is None:
            self.anchor = np.zeros(3)
        else:
            self.anchor = np.array(anchor, dtype=np.float64)
        # the name of the joint
        self.name = name

        # which links is this a joint between?
        self.connects = connects

        # the value to symbolically represent joint position
        self.parameter = sp.Symbol(name)

        self.initial = initial
        self.limits = limits

    @property
    def matrix(self):
        # inherit the docstring from the base class
        # self.parameter is a `sympy.Symbol` so the returned
        # transformation matrix will also be symbolic
        matrix = trimesh.transformations.rotation_matrix(
            angle=self.parameter,
            direction=self.axis,
            point=self.anchor)

        if self.initial is not None:
            matrix = matrix * sp.Matrix(self.initial)

        return matrix


class LinearJoint(Joint):
    def __init__(self, name, axis, connects, limits=None):
        """
        Create a linear (also known as prismatic) joint between
        two `Link` objects.

        Parameters
        -------------
        name : str
          The name of the joint
        axis : (3,) float
          The vector along which the joint translates
        connects : (2,) list
          Which links does the joint connect
        limits : None or (2,) float
          What are the limits of the joint
        """
        self.parameter = sp.Symbol(name)
        self.connects = connects

        self.limits = limits

        if axis is None or len(axis) != 3:
            raise ValueError('axis must be (3,) float!')

        # save axis as a unit vector
        self.axis = np.array(axis, dtype=np.float64)
        self.axis /= np.linalg.norm(self.axis)

    @property
    def matrix(self):
        """
        Get a parametrized transformation for this joint.

        Returns
        -----------
        matrix : (4, 4) sp.Matrix
          Transform parameterized by self.parameter
        """
        # start with an identity matrix
        translation = sp.Matrix.eye(4)
        # self.axis is a unit vector
        translation[:3, 3] = self.axis * self.parameter
        return translation
