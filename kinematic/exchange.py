"""
exchange.py
--------------

Load kinematic formats: OpenRAVE XML and URDF.
"""
import trimesh

import numpy as np

# for debugging
from trimesh.exchange.threedxml import print_element as pp  # NOQA

try:
    import lxml.etree as etree
except BaseException as E:
    etree = trimesh.exceptions.ExceptionModule(E)

from .link import Link
from .joint import Joint, RotaryJoint, LinearJoint
from .chain import KinematicChain


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
