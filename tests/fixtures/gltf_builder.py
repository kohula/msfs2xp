"""
Reusable synthetic glTF/GLB construction helper for the test suite.

Every fix verified throughout this project's development was checked with a
hand-rolled, throwaway script building a minimal binary GLB in memory (pad4/
chunk helpers, manual bufferView/accessor bookkeeping, ...). This module is
that same proven pattern, consolidated into one reusable builder so test
files don't each re-derive the GLB container format -- they just describe
the glTF content they need (a flat plane, an instanced mirrored mesh, a
hinge animation, a macro-light node, ...) and call build().

Deliberately NOT a general-purpose glTF authoring library: only the small
subset of the spec this project's own converter reads is implemented, and
only in the shapes the converter's own code expects (e.g. accessors always
get explicit min/max, matching what every real MSFS export -- and this
project's own prior ad-hoc test scripts -- already assumed).
"""

import json
import struct
import base64
import io
from typing import Optional, Sequence

import numpy as np

try:
    from PIL import Image
except ImportError:  # Pillow is a hard requirement of the real pipeline;
    Image = None      # tests that need add_image_data_uri will fail loudly if missing.

_GLTF_MAGIC = 0x46546C67
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942

_COMPONENT_TYPE_BYTES = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
_TYPE_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _pad4(data: bytes, fill: bytes = b"\x00") -> bytes:
    while len(data) % 4:
        data += fill
    return data


class GltfBuilder:
    def __init__(self):
        self._bin_chunks: list[bytes] = []
        self._buffer_views: list[dict] = []
        self._accessors: list[dict] = []
        self._meshes: list[dict] = []
        self._nodes: list[dict] = []
        self._materials: list[dict] = []
        self._images: list[dict] = []
        self._textures: list[dict] = []
        self._animations: list[dict] = []
        self._skins: list[dict] = []
        self._scene_nodes: list[int] = []
        self._extensions_used: set[str] = set()

    # -- low-level buffer/accessor plumbing -------------------------------

    def _add_buffer_view(self, raw: bytes) -> int:
        offset = sum(len(c) for c in self._bin_chunks)
        padded = _pad4(raw)
        self._bin_chunks.append(padded)
        idx = len(self._buffer_views)
        self._buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(raw)})
        return idx

    def add_accessor(self, data: np.ndarray, component_type: int, accessor_type: str) -> int:
        """data: numpy array, already in the right dtype/shape (N,) for
        SCALAR or (N, k) for VECk. Returns the new accessor's index."""
        flat = np.ascontiguousarray(data)
        bv_idx = self._add_buffer_view(flat.tobytes())
        count = flat.shape[0]
        ncomp = _TYPE_COMPONENTS[accessor_type]
        if ncomp == 1:
            mins = [float(flat.min())] if count else [0.0]
            maxs = [float(flat.max())] if count else [0.0]
        else:
            mins = flat.min(axis=0).tolist() if count else [0.0] * ncomp
            maxs = flat.max(axis=0).tolist() if count else [0.0] * ncomp
        idx = len(self._accessors)
        self._accessors.append({
            "bufferView": bv_idx, "componentType": component_type, "type": accessor_type,
            "count": count, "min": mins, "max": maxs,
        })
        return idx

    def add_positions(self, positions: Sequence[Sequence[float]]) -> int:
        return self.add_accessor(np.asarray(positions, dtype=np.float32), 5126, "VEC3")

    def add_normals(self, normals: Sequence[Sequence[float]]) -> int:
        return self.add_accessor(np.asarray(normals, dtype=np.float32), 5126, "VEC3")

    def add_uvs(self, uvs: Sequence[Sequence[float]]) -> int:
        return self.add_accessor(np.asarray(uvs, dtype=np.float32), 5126, "VEC2")

    def add_indices(self, indices: Sequence[int]) -> int:
        return self.add_accessor(np.asarray(indices, dtype=np.uint16), 5123, "SCALAR")

    # -- materials/textures -------------------------------------------------

    def add_image_uri(self, uri: str, name: Optional[str] = None) -> int:
        """Adds an image from an already-built URI (typically a data: URI
        shared verbatim across multiple separate GltfBuilder instances, so
        each one's converted output resolves to the identical texture
        name -- the real-world case of multiple placed objects genuinely
        sharing one texture asset)."""
        idx = len(self._images)
        entry = {"uri": uri}
        if name:
            entry["name"] = name
        self._images.append(entry)
        return idx

    def add_image_data_uri(self, rgba_color=(200, 200, 200, 255), size=(2, 2), name: Optional[str] = None) -> int:
        if Image is None:
            raise RuntimeError("Pillow not installed -- required to build a synthetic texture image")
        buf = io.BytesIO()
        Image.new("RGBA", size, rgba_color).save(buf, "PNG")
        uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        return self.add_image_uri(uri, name=name)

    @staticmethod
    def make_data_uri(rgba_color=(200, 200, 200, 255), size=(2, 2)) -> str:
        """Builds a data: URI without registering it on any builder --
        for sharing the identical URI string across multiple SEPARATE
        GltfBuilder instances via their own add_image_uri()."""
        if Image is None:
            raise RuntimeError("Pillow not installed -- required to build a synthetic texture image")
        buf = io.BytesIO()
        Image.new("RGBA", size, rgba_color).save(buf, "PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    def add_texture(self, image_index: int) -> int:
        idx = len(self._textures)
        self._textures.append({"source": image_index})
        return idx

    def add_material(self, name: str = "Mat", base_color_texture_index: Optional[int] = None,
                      emissive_texture_index: Optional[int] = None, extensions: Optional[dict] = None,
                      alpha_mode: Optional[str] = None, base_color_factor: Optional[list] = None) -> int:
        mat = {"name": name}
        pbr = {}
        if base_color_texture_index is not None:
            pbr["baseColorTexture"] = {"index": base_color_texture_index}
        if base_color_factor is not None:
            pbr["baseColorFactor"] = base_color_factor
        if pbr:
            mat["pbrMetallicRoughness"] = pbr
        if emissive_texture_index is not None:
            mat["emissiveTexture"] = {"index": emissive_texture_index}
        if alpha_mode:
            mat["alphaMode"] = alpha_mode
        if extensions:
            mat["extensions"] = extensions
            self._extensions_used.update(extensions.keys())
        idx = len(self._materials)
        self._materials.append(mat)
        return idx

    # -- meshes/nodes ---------------------------------------------------------

    def add_mesh(self, positions, indices, normals=None, uvs=None, material_index: Optional[int] = None) -> int:
        attrs = {"POSITION": self.add_positions(positions)}
        if normals is not None:
            attrs["NORMAL"] = self.add_normals(normals)
        if uvs is not None:
            attrs["TEXCOORD_0"] = self.add_uvs(uvs)
        prim = {"attributes": attrs, "indices": self.add_indices(indices)}
        if material_index is not None:
            prim["material"] = material_index
        idx = len(self._meshes)
        self._meshes.append({"primitives": [prim]})
        return idx

    def add_raw_mesh(self, primitives: list) -> int:
        """Escape hatch for primitive shapes add_mesh() doesn't cover --
        specifically multiple primitives (different materials) sharing ONE
        big POSITION/NORMAL/TEXCOORD accessor and each selecting only a
        small window of it via ASOBO_primitive's StartIndex/BaseVertexIndex/
        PrimitiveCount extras. This is a real, common MSFS export shape
        (one shared vertex buffer split across several material-specific
        draw calls); the caller builds each primitive dict directly via
        add_accessor()/add_indices() for whatever accessors it needs."""
        idx = len(self._meshes)
        self._meshes.append({"primitives": primitives})
        return idx

    def add_node(self, mesh_index: Optional[int] = None, name: Optional[str] = None,
                 children: Optional[Sequence[int]] = None, translation=None, rotation=None,
                 scale=None, extensions: Optional[dict] = None, skin_index: Optional[int] = None,
                 top_level: bool = True) -> int:
        node = {}
        if mesh_index is not None:
            node["mesh"] = mesh_index
        if name:
            node["name"] = name
        if children:
            node["children"] = list(children)
        if translation:
            node["translation"] = list(translation)
        if rotation:
            node["rotation"] = list(rotation)
        if scale:
            node["scale"] = list(scale)
        if skin_index is not None:
            node["skin"] = skin_index
        if extensions:
            node["extensions"] = extensions
            self._extensions_used.update(extensions.keys())
        idx = len(self._nodes)
        self._nodes.append(node)
        if top_level:
            self._scene_nodes.append(idx)
        return idx

    def add_instanced_node(self, mesh_index: int, name: str,
                            instance_translations: Sequence[Sequence[float]],
                            instance_rotations: Optional[Sequence[Sequence[float]]] = None,
                            instance_scales: Optional[Sequence[Sequence[float]]] = None,
                            top_level: bool = True) -> int:
        """EXT_mesh_gpu_instancing convenience: one node, N instances, each
        with its own TRANSLATION (and optionally ROTATION/SCALE) attribute
        accessor -- exactly the extension shape node_instance_matrices()
        reads in the real converter."""
        attrs = {"TRANSLATION": self.add_accessor(np.asarray(instance_translations, dtype=np.float32), 5126, "VEC3")}
        if instance_rotations is not None:
            attrs["ROTATION"] = self.add_accessor(np.asarray(instance_rotations, dtype=np.float32), 5126, "VEC4")
        if instance_scales is not None:
            attrs["SCALE"] = self.add_accessor(np.asarray(instance_scales, dtype=np.float32), 5126, "VEC3")
        return self.add_node(
            mesh_index=mesh_index, name=name,
            extensions={"EXT_mesh_gpu_instancing": {"attributes": attrs}},
            top_level=top_level,
        )

    def add_macro_light(self, node_index: int, color=(1.0, 1.0, 1.0), cone_angle=360.0, intensity=1.0,
                         day_night_cycle=False, flash_frequency=0.0):
        """Attaches ASOBO_macro_light to an already-added node in place."""
        ext = self._nodes[node_index].setdefault("extensions", {})
        ext["ASOBO_macro_light"] = {
            "color": list(color), "cone_angle": cone_angle, "intensity": intensity,
            "day_night_cycle": day_night_cycle, "flash_frequency": flash_frequency,
        }
        self._extensions_used.add("ASOBO_macro_light")

    def add_animation(self, target_node: int, path: str, times: Sequence[float], values: Sequence) -> int:
        """path: 'translation' | 'rotation' | 'scale'. values: Nx3 for
        translation/scale, Nx4 (quaternion xyzw) for rotation."""
        input_acc = self.add_accessor(np.asarray(times, dtype=np.float32), 5126, "SCALAR")
        vtype = "VEC4" if path == "rotation" else "VEC3"
        output_acc = self.add_accessor(np.asarray(values, dtype=np.float32), 5126, vtype)
        idx = len(self._animations)
        self._animations.append({
            "samplers": [{"input": input_acc, "output": output_acc}],
            "channels": [{"sampler": 0, "target": {"node": target_node, "path": path}}],
        })
        return idx

    def add_skin(self, joints: Sequence[int], inverse_bind_matrices: Optional[Sequence] = None) -> int:
        """inverse_bind_matrices: optional sequence of 4x4 nested
        lists/arrays, one per entry in `joints` (same order), mesh-local
        -> that joint's own space. Stored as glTF's own column-major MAT4
        accessor layout."""
        idx = len(self._skins)
        skin = {"joints": list(joints)}
        if inverse_bind_matrices is not None:
            flat = np.stack([np.asarray(m, dtype=np.float32).reshape(4, 4).T.flatten() for m in inverse_bind_matrices])
            skin["inverseBindMatrices"] = self.add_accessor(flat, 5126, "MAT4")
        self._skins.append(skin)
        return idx

    # -- assembly -------------------------------------------------------------

    def build(self) -> bytes:
        bin_chunk = b"".join(self._bin_chunks)
        gltf = {
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": self._scene_nodes}],
            "nodes": self._nodes,
            "meshes": self._meshes,
            "accessors": self._accessors,
            "bufferViews": self._buffer_views,
            "buffers": [{"byteLength": len(bin_chunk)}],
        }
        if self._materials:
            gltf["materials"] = self._materials
        if self._textures:
            gltf["textures"] = self._textures
        if self._images:
            gltf["images"] = self._images
        if self._animations:
            gltf["animations"] = self._animations
        if self._skins:
            gltf["skins"] = self._skins
        if self._extensions_used:
            gltf["extensionsUsed"] = sorted(self._extensions_used)

        json_bytes = _pad4(json.dumps(gltf).encode("utf-8"), b" ")

        def chunk(ctype, data):
            return struct.pack("<II", len(data), ctype) + data

        body = chunk(_CHUNK_JSON, json_bytes) + (chunk(_CHUNK_BIN, bin_chunk) if bin_chunk else b"")
        header = struct.pack("<III", _GLTF_MAGIC, 2, 12 + len(body))
        return header + body


def flat_quad(x0: float, x1: float, z0: float, z1: float, y: float = 0.0):
    """Standard CCW-from-above quad used by most fixtures: 4 verts, 2 tris,
    default UVs matching this project's own established convention
    (u increases with local +x, v increases with local +z)."""
    positions = [(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)]
    normals = [(0.0, 1.0, 0.0)] * 4
    uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    indices = [0, 1, 2, 0, 2, 3]
    return positions, normals, uvs, indices
