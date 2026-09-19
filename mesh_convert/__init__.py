"""
glTF -> X-Plane OBJ8 conversion package.

convert() is the stable public entry point (same signature and contract
as the previous single-file glb2obj.py's convert()): given a
source .glb/.gltf path and output directories, writes one or more OBJ8
.obj files and returns their paths.
"""

from .convert import convert

__all__ = ["convert"]
