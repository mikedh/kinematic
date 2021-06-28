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
import networkx as nx

ABC = trimesh.util.ABC
# for debugging
from trimesh.exchange.threedxml import print_element as pp  # NOQA

try:
    import lxml.etree as etree
except BaseException as E:
    etree = trimesh.exceptions.ExceptionModule(E)


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


def _parse_file(file_obj, ext):
    """
    Load an XML file from a file path or ZIP archive.

    Parameters
    ----------
    file_obj : str
      Path to an XML file or ZIP archive
    ext : str
      Desired extension of XML-like file

    Returns
    -----------
    tree : lxml.etree.Etr
    """
    # make sure extension is in the format '.extension'
    ext = '.' + ext.lower().strip().lstrip('.')

    # load our file into an etree and a resolver
    if trimesh.util.is_string(file_obj):
        if file_obj.lower().endswith(ext):
            # path was passed to actual XML file so resolver can use that path
            resolver = trimesh.visual.resolvers.FilePathResolver(file_obj)
            tree = etree.parse(file_obj)
        elif file_obj.lower().endswith('.zip'):
            # load the ZIP archive
            with open(file_obj, 'rb') as f:
                archive = trimesh.util.decompress(f, 'zip')
            # find the first key in the archive that matches our extension
            # this will be screwey if there are multiple XML files
            key = next(k for k in archive.keys()
                       if k.lower().endswith(ext))
            # load the XML file into an etree
            tree = etree.parse(archive[key])
            # create a resolver from the archive
            resolver = trimesh.visual.resolvers.ZipResolver(archive)
        else:
            raise ValueError(f'must be {ext} or ZIP with {ext} inside!')
    else:
        raise NotImplementedError('must load by file name')

    return tree, resolver


def _load_meshes(names, resolver):
    """
    Load a list of filenames from a resolver.

    Parameters
    -----------
    names : (n,) str
      List of file names
    resolver : trimesh.visual.Resolver
      Resolver which can load files

    Returns
    -----------
    meshes : dict
      Mesh name corresponding to mesh objects
    """
    meshes = {}
    for file_name in names:
        try:
            loaded = trimesh.load(
                file_obj=trimesh.util.wrap_as_stream(
                    resolver.get(file_name)),
                file_type=file_name)
        except BaseException as E:
            print('exception', E)
            continue

        if isinstance(loaded, trimesh.Trimesh):
            meshes[file_name] = loaded
        elif isinstance(loaded, trimesh.Scene):
            meshes.update(loaded.geometry)
        else:
            print('not a known geometry type!')

    return meshes


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

    tree, resolver = _parse_file(file_obj=file_obj, ext='.xml')

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
        # they reference the links on either side
        connects = [i.text for i in j.findall('Body')]
        # offset = j.find('offsetfrom').text
        # joint limits
        limits = np.array(j.find('limits').text.split(),
                          dtype=np.float64)
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
        geom = {}
        for g in b.iter('Geom'):
            if 'type' not in g.attrib:
                continue
            kind = g.attrib['type'].lower()
            if kind == 'trimesh':
                geom.update(_load_meshes(
                    names=[c.text for c in g.iter('{*}collision')],
                    resolver=resolver))
        # try setting the diffuse color
        color = g.find('diffuseColor')
        if color is not None:
            text = color.text.replace(',', ' ')
            color = np.array(text.split(),
                             dtype=np.float64)
            for mesh in geom.values():
                if hasattr(mesh.visual, 'face_colors'):
                    mesh.visual.face_colors = color
        if len(geom) == 0:
            return {}
        return {name: Link(name=name, geometry=geom)}

    links = {}
    joints = {}
    for robot in tree.iter('Robot'):
        for kinbody in robot.iter('KinBody'):
            for j in kinbody.findall('Joint'):
                joints.update(parse_joint(j))
            # only take the links that are children of kinbody
            for b in kinbody.findall('Body'):
                links.update(parse_body(b))
    chain = KinematicChain(joints=joints, links=links)

    return chain


def load_urdf(file_obj):
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

    def parse_origin(node):
        """
        Function mostly copied from `urdfpy`: find the `origin`
        subelement of an XML node and convert it into a homogenous
        transformation matrix.

        Parameters
        ----------
        node : :class`lxml.etree.Element`
            An XML node which optionally has an `origin` child node

        Returns
        -------
        matrix : (4, 4) float
          Transform matrix that corresponds to node origin child
          or identity matrix if no origin.
        """
        matrix = np.eye(4, dtype=np.float64)
        origin_node = node.find('origin')
        if origin_node is not None:
            if 'xyz' in origin_node.attrib:
                matrix[:3, 3] = np.fromstring(
                    origin_node.attrib['xyz'], sep=' ')
            if 'rpy' in origin_node.attrib:
                rpy = np.array(origin_node.attrib['rpy'].split(),
                               dtype=np.float64)
                # rpy notation is 'statix-zyz' in euler language
                matrix[:3, :3] = trimesh.transformations.euler_matrix(
                    *rpy, axes='szyx')[:3, :3]
        return matrix

    def parse_joint(j):
        if 'type' not in j.attrib:
            return {}
        name = j.attrib['name']
        kind = j.attrib['type']
        if kind != 'revolute':
            return {}
        connects = (j.find('{*}parent').attrib['link'],
                    j.find('{*}child').attrib['link'])
        # initial transform of the joint
        initial = parse_origin(j)
        anchor = initial[:3, 3]
        axis = np.array(j.find('axis').attrib['xyz'].split(),
                        dtype=np.float64)
        lim = j.find('limit')
        if lim is None:
            limits = None
        else:
            limits = np.array([lim.attrib['lower'], lim.attrib['upper']],
                              dtype=np.float64)
        return {name: RotaryJoint(name=name,
                                  axis=axis,
                                  anchor=anchor,
                                  limits=limits,
                                  initial=initial,
                                  connects=connects)}

    def parse_meshes(element):
        if element is None:
            return {}
        names = [m.attrib['filename']
                 for m in element.iter('{*}mesh')
                 if 'filename' in m.attrib]
        meshes = _load_meshes(names=names, resolver=resolver)
        return meshes

    def parse_link(L):
        name = L.attrib['name']
        meshes = parse_meshes(L.find('{*}visual'))
        body = Link(name=name,
                    geometry=meshes)
        return {name: body}

    tree, resolver = _parse_file(file_obj=file_obj, ext='.urdf')

    links = {}
    joints = {}
    for j in tree.iter('{*}joint'):
        joints.update(parse_joint(j))
    for L in tree.iter('{*}link'):
        links.update(parse_link(L))
    chain = KinematicChain(
        joints=joints, links=links, base_link='base_link')

    return chain


def show_bounce(chains, angle_per_step=0.01, **kwargs):
    """
    A basic demonstration which will visualize one or multiple
    KinematicChain objects bounding between their joint limits.

    Parameters
    -----------
    chains : dict or KinematicChain
      One or multiple chains
    kwargs : dict
      Passed to scene.show
    """
    def callback(scene):
        # use forward kinmatic functions to find new pose
        for chain_name, chain in chains.items():
            position = states[chain_name]
            for joint_name, F in forward[chain_name].items():
                matrix = F(*position)
                for geom_name in chain.links[joint_name].geometry.keys():
                    scene.graph.update(
                        frame_from=chain.base_frame,
                        frame_to=geom_name,
                        matrix=matrix)
            # reverse the motion direction when we hit a joint limit
            reverse = ((position <= limits[chain_name][:, 0]) |
                       (position >= limits[chain_name][:, 1]))
            direction = directions[chain_name]
            direction[reverse] *= -1
            position += np.ones(len(position)) * angle_per_step * direction
    # if passed a single chain put it into a dict
    if isinstance(chains, KinematicChain):
        chains = {'single': chains}

    # get a scene for every chain
    scenes = [c.scene() for c in chains.values()]
    # append them into one scene
    scene = trimesh.scene.scene.append_scenes(scenes)

    scene.camera_transform = [
        [0.9904557, 0., -0.13783142, 0.14432819],
        [-0.13766048, 0.04978808, -0.98922734, -2.04164079],
        [0.00686236, 0.9987598, 0.04931289, 0.48014882],
        [0., 0., 0., 1.]]

    # find an offset so each chain is right next to the others
    offset = np.append(0, np.cumsum(
        [s.extents[0] * 1.1 for s in scenes]))[:-1]

    for c, off in zip(chains.values(), offset):
        # offset the base frame of each chain by the desired distance
        scene.graph[c.base_frame] = trimesh.transformations.translation_matrix(
            [off, 0, 0])

    # variables which will be in-scope for callback
    forward = {k: c.forward_kinematics_lambda()
               for k, c in chains.items()}
    limits = {k: c.limits
              for k, c in chains.items()}
    directions = {k: -1 * np.ones(len(c.joints))
                  for k, c in chains.items()}
    states = {k: c.limits.mean(axis=1)
              for k, c in chains.items()}
    states['abb'] = np.zeros(6)
    states['ur5'] = np.ones(6) * -np.radians(90)

    # show the robot moving with a callback
    scene.show(callback=callback, record=True, **kwargs)

    return scene


if __name__ == '__main__':

    from pyinstrument import Profiler
    profiler = Profiler()
    profiler.start()

    abb = load_orxml('robots/irb140.zip')
    ur5 = load_urdf('robots/ur5.zip')

    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))

    r = show_bounce({'abb': abb, 'ur5': ur5})
