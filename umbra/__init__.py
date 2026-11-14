"""umbra - 2.5D silhouette-and-diorama animation, built for AI agents. Runs on Blender (EEVEE, GPU).

    from umbra import *

    class Hello(Scene):
        def build(s):
            ...
"""
__version__ = "0.1.0"

try:
    import bpy as _bpy  # noqa: F401
    IN_BLENDER = True
except ImportError:            # plain Python: only the CLI is usable
    IN_BLENDER = False

from .colors import *          # noqa: F401,F403  (palette names)
from .colors import lin, srgb, mix as mix_color, PALETTE, BG
from .track import ease, EASES, lerp, clamp, smoothstep, noise1
from .xform import v3, Path, yaw_to

if IN_BLENDER:
    from . import geom
    from .geom import Mesh, box, cyl, cone, sphere, rock, plane, extrude, lathe, tube, terrain
    from .look import Look, NOIR, CLAY, BLUEPRINT, DUSK, PAPER, LOOKS
    from .node import Node, LightNode
    from .scene import Scene
