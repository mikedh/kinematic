"""
kinematic.py
--------------

PROTOTYPE for support for rigid body kinematics. Not sure if
supporting this is something that is feasible or desirable for trimesh,
this is just exploring what it might look like. If this improves, it
could feasibly live as `trimesh.kinematic`.

Challenge: create a data structure which can (mostly) hold and
cross-convert GLTF Skeletons, OpenRave XML Robots, and URDF robots.

Also, trimesh scenes aren't the greatest thing in the world. Maybe they would
need a refactor?
"""

import trimesh

import abc
import sympy as sp
import numpy as np
import networkx as nx

# for debugging
from trimesh.exchange.xml_based import print_element as pp  # NOQA


class KinematicChain(object):
    """
    A mechanism which consists of geometry (`Body` objects) connected
    by transforms (`Joint` objects).
    """

    def __init__(self,
                 joints,
                 bodies,
                 base_body='base'):
        """
        Create a kinematic chain.

        Parameters
        --------------
        joints : (n,) Joint objects
            Connections between Bodies
        bodies : (m,) Body object
            Geometryy
        """
        self.joints = joints

        # which body is the first
        self.base_body = base_body
        self.bodies = bodies

    @property
    def parameters(self):
        return [i.parameter for i in self.joints.values()]

    @property
    def limits(self):
        limits = np.array([j.limits for j in self.joints.values()])
        limits.sort(axis=1)
        return limits

    def graph(self):
        graph = nx.DiGraph()
        for name, joint in self.joints.items():
            graph.add_edge(*joint.connects, joint=name)
        return graph

    def scene(self):
        geometries = {}
        for name, body in self.bodies.items():
            geom = body.geometries
            if len(geom) == 1:
                geometries[name] = geom[0]
        return trimesh.Scene(geometries)

    def paths(self, base='base'):
        graph = self.graph()
        paths = {b.name: nx.shortest_path(graph, base, b.name)
                 for b in self.bodies.values()}

        joints = {}
        for body, path in paths.items():
            joints[body] = [graph.get_edge_data(a, b)['joint']
                            for a, b in zip(path[:-1], path[1:])]
        return joints

    def transforms_symbolic(self):
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

    def transforms(self):
        """
        Get a numpy-lambda for evaluating forward kinematics relatively
        quickly.

        Returns
        -----------
        lambdas : dict
          Body name to function which takes float values
          corresponding to self.parameters.
        """
        combined = self.transforms_symbolic()
        return {k: sp.lambdify(self.parameters, c)
                for k, c in combined.items()}


def make_immutable(obj):
    """
    Make an object immutable-ish.

    TODO : make most values of kinematic chains / joints / bodies
    immutable so we can precompute stuff in the __init__
    """
    def immutable(*args, **kwargs):
        raise ValueError('object is immutable!')
    obj.__setattr__ = immutable


class Joint(trimesh.util.ABC):
    @abc.abstractmethod
    def matrix(self):
        raise NotImplementedError('call a subclass!')

    @abc.abstractproperty
    def connects(self):
        raise NotImplementedError('call a subclass!')


class RotaryJoint(Joint):
    def __init__(self,
                 axis,
                 connects,
                 limits=None,
                 anchor=None,
                 name=None):
        """
        Create a rotary joint.
        """
        # the unit vector axis
        self.axis = np.asanyarray(axis, dtype=np.float64).reshape(3)
        # the point around which to rotate
        if anchor is None:
            self.anchor = np.zeros(3)
        else:
            self.anchor = np.asanyarray(anchor, dtype=np.float64)
        # the name of the joint
        self.name = name

        # which bodies is this a joint between?
        self.connects = connects

        # the value to symbolically represent joint position
        self.parameter = sp.Symbol(name)

        self.limits = limits

    @property
    def matrix(self):
        """
        Return a symbolic transformation matrix for the joint.

        Returns
        ------------
        matrix : sympy.Matrix
          Transform with self.parameter as variable
        """
        matrix = trimesh.transformations.rotation_matrix(
            angle=self.parameter,
            direction=self.axis,
            point=self.anchor)
        return matrix

    @property
    def connects(self):
        return self._connects

    @connects.setter
    def connects(self, values):
        if len(values) != 2:
            raise ValueError()
        self._connects = values


class Body(object):
    def __init__(self, name, geometries):
        self.name = name
        self.geometries = geometries

    def show(self, **kwargs):
        trimesh.Scene(self.geometries).show(**kwargs)


def load_orxml(file_obj):
    """
    Load an OpenRAVE XML file from a ZIP or file path.

    Parameters
    ------------
    file_obj : str
      Path to XML file or ZIP with XML inside

    Returns
    ------------
    chain : KinematicChain
      Loaded result from XML
    """

    import lxml
    # load our file into an etree and a resolver
    if trimesh.util.is_string(file_obj):
        if file_obj.lower().endswith('.xml'):
            resolver = trimesh.visual.resolvers.FilePathResolver(file_obj)
            tree = lxml.etree.parse(file_obj)
        elif file_obj.lower().endswith('zip'):
            with open(file_obj, 'rb') as f:
                archive = trimesh.util.decompress(f, 'zip')
            key = next(k for k in archive.keys() if k.lower().endswith('.xml'))
            tree = lxml.etree.parse(archive[key])

            resolver = trimesh.visual.resolvers.ZipResolver(archive)
        else:
            raise ValueError('must be XML or ZIP with XML!')
    else:
        raise NotImplementedError('load by filename')

    def parse_child(element, name):
        name = str(name).strip().lower()
        for child in element:
            if child.tag.strip().lower() != name:
                continue
            return np.array(child.text.split(), dtype=np.float64)
        raise ValueError('no such child')

    def parse_joint(j):
        axis = parse_child(j, 'axis')
        try:
            anchor = parse_child(j, 'anchor')
        except ValueError:
            anchor = [0, 0, 0]
        name = j.attrib['name']

        # they reference the bodies on either side
        connects = [i.text for i in j.findall('Body')]

        # offset = j.find('offsetfrom').text
        # joint limits
        limits = np.array(j.find('limits').text.split(), dtype=np.float64)

        # pp(j)

        return {name: RotaryJoint(
            axis=axis,
            anchor=anchor,
            connects=connects,
            limits=limits,
            name=name)}

    def parse_body(b):
        if 'name' not in b.attrib:
            return {}
        name = b.attrib['name']
        geom = []
        for g in b.iter('Geom'):
            if 'type' not in g.attrib:
                continue
            kind = g.attrib['type']
            if kind == 'trimesh':
                loaded = None
                try:
                    loaded = [trimesh.load(
                        file_obj=trimesh.util.wrap_as_stream(
                            resolver.get(c.text)),
                        file_type=c.text)
                        for c in g.iter('collision')]
                except BaseException as E:
                    print('failed', E)
                    continue
                try:
                    # try setting the diffuse color
                    color = g.find('diffuseColor')
                    if color is not None:
                        text = color.text.replace(',', ' ')
                        color = np.array(text.split(),
                                         dtype=np.float64)
                        for i in loaded:
                            i.visual.face_colors = color
                except BaseException:
                    pass
                if loaded is not None:
                    geom.extend(loaded)
        if len(geom) == 0:
            return {}
        return {name: Body(name=name, geometries=geom)}

    bodies = {}
    joints = {}
    for robot in tree.iter('Robot'):
        for kinbody in robot.iter('KinBody'):
            for j in kinbody.findall('Joint'):
                joints.update(parse_joint(j))
            # only take the bodies that are children of kinbody
            for b in kinbody.findall('Body'):
                bodies.update(parse_body(b))
    chain = KinematicChain(joints=joints, bodies=bodies)

    return chain


def callback(scene):
    """
    A basic callback which moves the arm between it's joint limits.
    """
    pos = scene.state
    # use forward kinmatic functions to find new pose
    for name, F in forward.items():
        scene.graph[name] = F(*pos)
    # reverse the motion direction when we hit a joint limit
    rev = (pos <= limits[:, 0]) | (pos >= limits[:, 1])
    scene.direction[rev] *= -1

    scene.state += np.ones(6) * .01 * scene.direction


if __name__ == '__main__':
    chain = load_orxml('robots/irb140.zip')

    # get a scene containing the chain
    scene = chain.scene()

    # numpy lambdas for forward kinematics
    forward = chain.transforms()

    # joint limits
    limits = chain.limits
    # store the direction of motion for the joints
    scene.direction = np.ones(6)
    # save the joint position to the scene
    scene.state = np.zeros(6)
    # show the robot moving with a callback
    scene.show(callback=callback)
