
import trimesh
import kinematic

import numpy as np


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

    # get a scene for every chain
    scenes = [c.scene() for c in chains.values()]
    # append them into one scene
    scene = trimesh.scene.scene.append_scenes(scenes)

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
    directions = {k: np.ones(len(c.joints))
                  for k, c in chains.items()}
    states = {k: np.zeros(len(c.joints))
              for k, c in chains.items()}

    # show the robot moving with a callback
    scene.show(callback=callback, **kwargs)


if __name__ == '__main__':

    a = kinematic.exchange.load_orxml('robots/irb140.zip')
    u = kinematic.exchange.load_urdf('robots/ur5.zip')

    show_bounce({'abb': a, 'ur5': u})
