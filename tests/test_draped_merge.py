"""
draped_merge.py rewritten to consume MeshIR sidecars instead of
regex-parsing OBJ8 text, and geo_transform.py instead of its own
duplicated rotation math. Pins vertex welding, correct combined footprint
positioning, different-texture isolation, and the exclusion rules
(blink/animated, rigid, AGL-mounted, library-substitution entries never
get merged).
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from dsf_fixture import build_elevation_dsf  # noqa: E402
from gltf_builder import GltfBuilder  # noqa: E402

import draped_merge
import geo_transform
import mesh_convert
import pol_writer
import terrain_fit
from mesh_convert import mesh_ir


def _placement_entry(obj_dir, model_stem, obj_stem, base_lat, base_lon, hdg=0.0, agl=0.0):
    """convert() now re-centers every model around its own XZ footprint
    and writes the removed (x, z) offset (plus any Y lift folded into agl)
    to a "<model_stem>.originoffset.json" sidecar -- mirrors what main.py
    does with it: rotate the offset by heading and fold it into the
    placement's own lat/lon/agl, so two independently re-centered files
    that were adjacent/coincident in their ORIGINAL local space land back
    in the same real-world position they would have without re-centering."""
    sidecar = obj_dir / f"{model_stem}.originoffset.json"
    ox = oz = oy = 0.0
    if sidecar.exists():
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        ox, oy, oz = data["x"], data["y"], data["z"]
    lat, lon = geo_transform.local_offset_to_latlon(base_lat, base_lon, hdg, ox, oz)
    return {"name": obj_stem, "lat": lat, "lon": lon, "hdg": hdg, "agl": agl + oy}


def _shared_texture_data_uri():
    return GltfBuilder.make_data_uri((120, 120, 120, 255))


def _idx_from_obj8(text):
    """The IDX stream of a written OBJ8 as an int64 array -- so a test can
    re-run draped_merge's own naked-edge/loop analysis on a merge result."""
    return np.array([int(l.split()[1]) for l in text.splitlines() if l.startswith("IDX ")],
                    dtype=np.int64)


def _make_quad_glb(path: Path, name: str, data_uri: str, tex_name: str, x0, x1, z0, z1):
    b = GltfBuilder()
    imgi = b.add_image_uri(data_uri, name=tex_name)
    texi = b.add_texture(imgi)
    mat = b.add_material(f"{name}Mat", base_color_texture_index=texi)
    positions = [(x0, 0, z0), (x1, 0, z0), (x1, 0, z1), (x0, 0, z1)]
    normals = [(0.0, 1.0, 0.0)] * 4
    uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    indices = [0, 1, 2, 0, 2, 3]
    mesh = b.add_mesh(positions, indices, normals=normals, uvs=uvs, material_index=mat)
    b.add_node(mesh_index=mesh, name=name)
    path.write_bytes(b.build())


class TestDrapedMerge(unittest.TestCase):
    def test_two_adjacent_same_texture_tiles_weld_and_merge(self):
        data_uri = _shared_texture_data_uri()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            glbA = td / "tileA.glb"
            _make_quad_glb(glbA, "TileA", data_uri, "SharedAsphalt", 0, 10, -10, 0)
            glbB = td / "tileB.glb"
            _make_quad_glb(glbB, "TileB", data_uri, "SharedAsphalt", 0, 10, 0, 10)

            resultA = mesh_convert.convert(glbA, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            resultB = mesh_convert.convert(glbB, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            stemA, stemB = resultA[0].stem, resultB[0].stem

            base_lat, base_lon = 47.0, 8.0
            entries = [
                _placement_entry(obj_dir, "tileA", stemA, base_lat, base_lon),
                _placement_entry(obj_dir, "tileB", stemB, base_lat, base_lon),
            ]
            tile_lat, tile_lon = 47, 8
            result, _ = draped_merge.merge_draped_layers_in_tile(obj_dir, tile_lat, tile_lon, entries)

            # Two same-texture OPAQUE pavement quads that butt exactly (gap 0)
            # pool into one draw call AND get their shared-edge boundary verts
            # INDEX-welded (_weld_component_gaps) -- so the two tiles become
            # one connected mesh with no inter-component edge for X-Plane's
            # draped renderer to crack. 8 raw verts -> 6 unique (the 2+2 verts
            # on the z=0 seam collapse to 2). No _seam_bridge object.
            non_bridge = [e for e in result if not e["name"].startswith("_seam_bridge_tile_")]
            self.assertEqual(len(non_bridge), 1)
            merged_name = non_bridge[0]["name"]
            self.assertTrue(merged_name.startswith("_merged_"))
            self.assertFalse([e for e in result if e["name"].startswith("_seam_bridge_tile_")],
                             "an exact butt joint has no gap to fill")

            merged_text = (obj_dir / f"{merged_name}.obj").read_text(encoding="utf-8")
            vt_lines = [l for l in merged_text.splitlines() if l.startswith("VT ")]
            self.assertEqual(len(vt_lines), 6, "shared-edge boundary verts weld: 8 -> 6")

            xs = [float(l.split()[1]) for l in vt_lines]
            zs = [float(l.split()[3]) for l in vt_lines]
            self.assertAlmostEqual(max(xs) - min(xs), 10.0, places=2)
            self.assertAlmostEqual(max(zs) - min(zs), 20.0, places=2)

            idx_count = sum(1 for l in merged_text.splitlines() if l.startswith("IDX "))
            self.assertEqual(idx_count, 12)
            self.assertIn("TRIS 0 12", merged_text)

    def test_terrain_fit_then_weld_still_merges_cleanly(self):
        """The confirmed real regression: main.py runs terrain_fit BEFORE
        draped_merge (terrain-correct each placement's own geometry
        independently, THEN weld/merge adjacent same-texture pieces
        across placements). If terrain_fit warped each placement's Y using
        a grid interpolated within its OWN local footprint bbox, two
        adjacent pieces' shared boundary vertices came out with slightly
        different Y (different bboxes -> different interpolation) -- just
        barely enough to push them outside _weld_vertices' tolerance:
        pieces that used to weld cleanly stopped welding, and duplicate
        layers stopped deduplicating. terrain_fit.py's current per-vertex-
        exact sampling fixes this at the root: the same real-world
        boundary point always gets the identical elevation lookup
        regardless of which placement it came from. This runs BOTH passes
        in the real pipeline order and confirms two independently-warped
        adjacent tiles still weld down to exactly the same 6 unique
        vertices as the unwarped case."""
        data_uri = _shared_texture_data_uri()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            xplane_root = td / "XPlaneRoot"
            obj_dir.mkdir()
            tex_dir.mkdir()

            # A real, spatially-varying (not flat) elevation surface across
            # the whole default-scenery tile -- if terrain_fit found
            # nothing to correct, this test couldn't tell a real per-vertex
            # fix apart from a no-op.
            width = height = 5
            slope_per_post = 200.0
            grid = [[(row + col) * slope_per_post for col in range(width)] for row in range(height)]
            dsf_dir = xplane_root / "Global Scenery" / "X-Plane 12 Global Scenery" / "Earth nav data" / "+40+000"
            dsf_dir.mkdir(parents=True, exist_ok=True)
            (dsf_dir / "+47+008.dsf").write_bytes(build_elevation_dsf(grid))

            # 20x20 each (not 10x10) -- must clear terrain_fit's own
            # _MIN_SIDE_M/_AREA_THRESHOLD_M2 qualification.
            glbA = td / "tileA.glb"
            _make_quad_glb(glbA, "TileA", data_uri, "SharedAsphalt", 0, 20, -20, 0)
            glbB = td / "tileB.glb"
            _make_quad_glb(glbB, "TileB", data_uri, "SharedAsphalt", 0, 20, 0, 20)

            resultA = mesh_convert.convert(glbA, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            resultB = mesh_convert.convert(glbB, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            stemA, stemB = resultA[0].stem, resultB[0].stem

            # Same real-world placement point for both (matching the
            # adjacent-tile weld test above -- this is what two real
            # adjacent MSFS placements sharing a reference point converts
            # to). The regression isn't about the placements differing --
            # it's that terrain_fit is called ONCE PER PLACEMENT,
            # independently (main.py's real pipeline order, before
            # draped_merge ever sees them together), and each call's own
            # local footprint bbox differs (tileA spans z in [-20,0],
            # tileB spans z in [0,20]) even at an identical base point --
            # exactly the case where a shared-bbox grid used to produce
            # two different interpolated Y values for what is really the
            # same real-world boundary edge.
            base_lat_a = base_lat_b = 47.0
            base_lon_a = base_lon_b = 8.0
            fit_a = terrain_fit.get_or_create_fitted_group(obj_dir, [stemA], base_lat_a, base_lon_a, 0.0, xplane_root)
            fit_b = terrain_fit.get_or_create_fitted_group(obj_dir, [stemB], base_lat_b, base_lon_b, 0.0, xplane_root)
            fitted_stem_a, applied_a, reason_a = fit_a[stemA]
            fitted_stem_b, applied_b, reason_b = fit_b[stemB]
            self.assertTrue(applied_a, f"test setup issue: expected a real correction (reason={reason_a!r})")
            self.assertTrue(applied_b, f"test setup issue: expected a real correction (reason={reason_b!r})")

            entries = [
                _placement_entry(obj_dir, "tileA", fitted_stem_a, base_lat_a, base_lon_a),
                _placement_entry(obj_dir, "tileB", fitted_stem_b, base_lat_b, base_lon_b),
            ]
            result, _ = draped_merge.merge_draped_layers_in_tile(obj_dir, 47, 8, entries)

            non_bridge = [e for e in result if not e["name"].startswith("_seam_bridge_tile_")]
            self.assertEqual(len(non_bridge), 1,
                             "two independently terrain-fit-warped adjacent tiles must still pool into one")
            merged_text = (obj_dir / f"{non_bridge[0]['name']}.obj").read_text(encoding="utf-8")
            vt_lines = [l for l in merged_text.splitlines() if l.startswith("VT ")]
            # terrain_fit must give the SAME real-world boundary point the SAME
            # elevation from either placement, so the two tiles' shared edge
            # lands COINCIDENT -- and _weld_component_gaps then index-merges it
            # (8 -> 6 verts). If terrain_fit had produced two different Y for
            # the shared edge, the weld's eps would miss and it would stay 8.
            self.assertEqual(len(vt_lines), 6, "shared edge coincident -> boundary verts weld (8 -> 6)")
            pts = sorted(tuple(round(float(v), 4) for v in l.split()[1:4]) for l in vt_lines)
            boundary = [p for p in pts if abs(p[2]) < 1e-3]  # z ~ 0, the shared edge
            self.assertEqual(len(boundary), 2, "the 2+2 shared-edge verts merged to 2")

    def test_merge_carries_blend_and_double_sided_through_from_members(self):
        """Confirmed real gap: the merged MeshIR built by
        merge_draped_layers_in_tile used to never set alpha_mode/
        double_sided/alpha_cutoff at all, silently discarding whatever the
        ORIGINAL members had authored -- contradicts this module's own
        top-of-file docstring intent (welding/dedup only, never changing
        appearance). One BLEND, double-sided member sharing a texture with
        an otherwise-OPAQUE, single-sided sibling must make the WHOLE
        merged result BLEND + double-sided, since a merged mesh can't
        selectively blend/cull only part of itself."""
        with tempfile.TemporaryDirectory() as td:
            obj_dir = Path(td)

            opaque_ir = mesh_ir.MeshIR(
                name="opaque_part", texture="../textures/shared.png", draped=True, draped_layer_offset=-2,
                positions=np.array([[0, 0, 0], [10, 0, 0], [10, 0, 10], [0, 0, 10]], dtype=np.float64),
                normals=np.array([[0, 1, 0]] * 4, dtype=np.float64),
                uvs=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64),
                indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.int64),
                alpha_mode="OPAQUE", double_sided=False,
            )
            mesh_ir.save(opaque_ir, mesh_ir.sidecar_path_for(obj_dir / "opaque_part.obj"))
            mesh_ir.write_obj8(opaque_ir, obj_dir / "opaque_part.obj")

            blend_ir = mesh_ir.MeshIR(
                name="blend_part", texture="../textures/shared.png", draped=True, draped_layer_offset=-2,
                positions=np.array([[20, 0, 0], [30, 0, 0], [30, 0, 10], [20, 0, 10]], dtype=np.float64),
                normals=np.array([[0, 1, 0]] * 4, dtype=np.float64),
                uvs=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64),
                indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.int64),
                alpha_mode="BLEND", double_sided=True,
            )
            mesh_ir.save(blend_ir, mesh_ir.sidecar_path_for(obj_dir / "blend_part.obj"))
            mesh_ir.write_obj8(blend_ir, obj_dir / "blend_part.obj")

            entries = [
                {"name": "opaque_part", "lat": 47.0, "lon": 8.0, "hdg": 0.0, "agl": 0.0},
                {"name": "blend_part", "lat": 47.0, "lon": 8.0, "hdg": 0.0, "agl": 0.0},
            ]
            result, _ = draped_merge.merge_draped_layers_in_tile(obj_dir, 47, 8, entries)
            self.assertEqual(len(result), 1, "same-texture members must still merge into one object")
            merged_text = (obj_dir / f"{result[0]['name']}.obj").read_text(encoding="utf-8")
            self.assertIn("ATTR_blend", merged_text, "one BLEND member must make the whole merged result BLEND")
            self.assertIn("ATTR_no_cull", merged_text, "one double-sided member must make the whole merged result double-sided")

    def test_fully_overlapping_same_texture_duplicate_is_removed_not_just_welded(self):
        """The confirmed real bug: a base apron polygon placed TWICE at the
        exact same real-world footprint with the same texture (e.g. picked
        up from two overlapping placement records) -- welding alone snaps
        both copies' vertices onto the same 4 shared positions, but without
        face-level dedup the merged mesh still carries BOTH copies'
        triangles referencing those same 4 vertices, which is exactly the
        z-fighting/"hole in the pavement" symptom reported. Two fully
        duplicate quads (4 unique welded vertices, 2 triangles each) must
        collapse to just 4 vertices and 2 triangles total, not 4."""
        data_uri = _shared_texture_data_uri()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            glbA = td / "apronA.glb"
            _make_quad_glb(glbA, "ApronA", data_uri, "SharedAsphalt", 0, 10, 0, 10)
            glbB = td / "apronB.glb"
            _make_quad_glb(glbB, "ApronB", data_uri, "SharedAsphalt", 0, 10, 0, 10)

            resultA = mesh_convert.convert(glbA, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            resultB = mesh_convert.convert(glbB, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            stemA, stemB = resultA[0].stem, resultB[0].stem

            base_lat, base_lon = 47.0, 8.0
            entries = [
                _placement_entry(obj_dir, "apronA", stemA, base_lat, base_lon),
                _placement_entry(obj_dir, "apronB", stemB, base_lat, base_lon),
            ]
            result, _ = draped_merge.merge_draped_layers_in_tile(obj_dir, 47, 8, entries)

            self.assertEqual(len(result), 1)
            merged_name = result[0]["name"]
            merged_text = (obj_dir / f"{merged_name}.obj").read_text(encoding="utf-8")

            vt_lines = [l for l in merged_text.splitlines() if l.startswith("VT ")]
            self.assertEqual(len(vt_lines), 4, "two fully-overlapping quads must weld to 4 unique vertices")

            idx_count = sum(1 for l in merged_text.splitlines() if l.startswith("IDX "))
            self.assertEqual(idx_count, 6, "the duplicate quad's 2 triangles must be dropped, not kept alongside the original's")
            self.assertIn("TRIS 0 6", merged_text)

    def test_merged_footprint_area_is_real_coverage_not_scattered_bounding_box(self):
        """Confirmed real regression against a real LHBP conversion: a
        shared dirt/wear decal texture reused on dozens of small, scattered
        objects (doors, props, tire marks) all over a real airport welds
        into ONE merge group whose bounding box spans nearly the airport's
        FULL extent, even though each individual patch covers only a
        couple square meters. Using that bounding box as "footprint_area"
        made every such scattered-decal group look astronomically large
        (100,000+ sq m), landing it in the exact same bottom draw-order
        bucket as the real base pavement fills -- the confirmed real
        symptom, every merged draped layer in a real tile coming back with
        the identical draped_layer_offset=-5, no real distinction between
        paint markings/decals/pavement at all. Two small (1msq each) same-
        texture quads placed 1km apart must weld into a group whose
        footprint_area is close to their small REAL combined area (~2 sq
        m), not anywhere near the ~1,000,000 sq m their separation would
        imply as a bounding box."""
        data_uri = _shared_texture_data_uri()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            glbA = td / "decalA.glb"
            _make_quad_glb(glbA, "DecalA", data_uri, "SharedDirt", 0, 1, 0, 1)
            glbB = td / "decalB.glb"
            _make_quad_glb(glbB, "DecalB", data_uri, "SharedDirt", 0, 1, 0, 1)

            stemA = mesh_convert.convert(glbA, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")[0].stem
            stemB = mesh_convert.convert(glbB, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")[0].stem

            # Same texture, but placed ~1km apart -- like two unrelated
            # scattered decals sharing one generic wear-atlas texture.
            entries = [
                _placement_entry(obj_dir, "decalA", stemA, 47.0, 8.0),
                _placement_entry(obj_dir, "decalB", stemB, 47.009, 8.0),  # ~1km north
            ]
            tile_lat, tile_lon = 47, 8
            result, _ = draped_merge.merge_draped_layers_in_tile(obj_dir, tile_lat, tile_lon, entries)

            self.assertEqual(len(result), 1, "same-texture pieces still merge into one object despite the distance")
            area = result[0]["footprint_area"]
            self.assertLess(area, 10.0,
                             f"footprint_area must reflect real small coverage (~2 sq m), not the "
                             f"~1,000,000 sq m bounding box implied by the 1km separation, got {area}")

    def test_merged_placement_is_recentered_not_left_at_the_tile_floor(self):
        """Confirmed real regression against a real LHBP conversion: a
        merged pavement patch sitting near the real airport (~0.43 deg
        north, ~0.25 deg east of its own DSF tile's floor/SW corner --
        tile_lat/tile_lon are math.floor() of the real coordinates, not
        anywhere near a real airport dead-center in its tile) got written
        with placement lat/lon EQUAL to the raw tile floor and its local
        VT coordinates left at their full tile-relative magnitude -- tens
        of thousands of meters, confirmed as far as 15-23km in X and
        47-50km in Z in the real case -- instead of the few tens of
        meters an OBJ8's local geometry is supposed to span relative to
        its OWN placement anchor. Mirrors this same module's own real
        member positions (_member_point_to_tile_frame) far from the tile
        floor, exactly like the real airport case."""
        data_uri = _shared_texture_data_uri()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            glbA = td / "tileA.glb"
            _make_quad_glb(glbA, "TileA", data_uri, "SharedAsphalt", 0, 10, -10, 0)
            glbB = td / "tileB.glb"
            _make_quad_glb(glbB, "TileB", data_uri, "SharedAsphalt", 0, 10, 0, 10)

            resultA = mesh_convert.convert(glbA, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            resultB = mesh_convert.convert(glbB, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            stemA, stemB = resultA[0].stem, resultB[0].stem

            # Real placement far from its own tile's floor corner -- same
            # shape as the real LHBP case (tile floor 47,19; real airport
            # at ~47.43,19.25).
            real_lat, real_lon = 47.43, 8.25
            tile_lat, tile_lon = 47, 8
            entries = [
                _placement_entry(obj_dir, "tileA", stemA, real_lat, real_lon),
                _placement_entry(obj_dir, "tileB", stemB, real_lat, real_lon),
            ]
            result, _ = draped_merge.merge_draped_layers_in_tile(obj_dir, tile_lat, tile_lon, entries)

            self.assertEqual(len(result), 1)
            entry = result[0]
            # The placement anchor must land close to the real airport
            # location, not sit unmoved at the raw tile floor.
            self.assertAlmostEqual(entry["lat"], real_lat, places=2)
            self.assertAlmostEqual(entry["lon"], real_lon, places=2)
            dist_from_floor_deg = ((entry["lat"] - tile_lat) ** 2 + (entry["lon"] - tile_lon) ** 2) ** 0.5
            self.assertGreater(dist_from_floor_deg, 0.1, "test setup issue: anchor should have moved from the tile floor")

            # And the LOCAL geometry itself must be small (a real placed
            # object's own footprint), not tens of thousands of meters --
            # the actual, confirmed symptom (pavement rendering nowhere
            # near where it should) came from X-Plane trying to render an
            # OBJ8 whose own local vertices were tens of km from its
            # placement anchor.
            merged_text = (obj_dir / f"{entry['name']}.obj").read_text(encoding="utf-8")
            vt_lines = [l for l in merged_text.splitlines() if l.startswith("VT ")]
            xs = [float(l.split()[1]) for l in vt_lines]
            zs = [float(l.split()[3]) for l in vt_lines]
            self.assertLess(max(abs(v) for v in xs + zs), 100.0,
                             "merged local geometry must stay near its own footprint, not span tile-relative kilometers")

    def test_merged_layer_is_not_east_west_stretched_across_km(self):
        """Two same-texture pieces ~3 km apart east/west, merged. Rebuilding
        each piece's real-world position the way X-Plane will -- placement
        anchor + local metres on a WGS84 tangent plane at the anchor --
        must land within 0.15 m of where each piece was actually placed.

        The old frame took local->lat/lon scaled by cos(member_lat) then
        lat/lon->tile metres scaled by cos(FLOOR(member_lat)); those two
        latitudes are ~0.5 deg apart at LHBP, a ~0.47% east/west stretch
        that put the far piece of a pooled taxi-line / marking layer ~7 m
        off the geodetically-placed default markings."""
        import geo_transform
        with tempfile.TemporaryDirectory() as td:
            obj_dir = Path(td)
            tile_lat, tile_lon = 47, 19
            plat = 47.43
            _, m_lon_ref = geo_transform.metres_per_degree(plat)
            lonA = 19.24
            lonB = lonA + 3000.0 / m_lon_ref  # ~3 km east

            def _unit_quad(name, lat, lon):
                ir = mesh_ir.MeshIR(
                    name=name, texture="../textures/gp_lines.png", draped=True, draped_layer_offset=-2,
                    positions=np.array([[-0.5, 0, -0.5], [0.5, 0, -0.5], [0.5, 0, 0.5], [-0.5, 0, 0.5]],
                                       dtype=np.float64),
                    normals=np.array([[0, 1, 0]] * 4, dtype=np.float64),
                    uvs=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64),
                    indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.int64),
                )
                mesh_ir.save(ir, mesh_ir.sidecar_path_for(obj_dir / f"{name}.obj"))
                mesh_ir.write_obj8(ir, obj_dir / f"{name}.obj")
                return {"name": name, "lat": lat, "lon": lon, "hdg": 0.0, "agl": 0.0}

            entries = [_unit_quad("pieceA", plat, lonA), _unit_quad("pieceB", plat, lonB)]
            result, _ = draped_merge.merge_draped_layers_in_tile(obj_dir, tile_lat, tile_lon, entries)
            self.assertEqual(len(result), 1)
            e = result[0]
            text = (obj_dir / f"{e['name']}.obj").read_text(encoding="utf-8")
            vt = [l.split() for l in text.splitlines() if l.startswith("VT ")]
            xs = [float(v[1]) for v in vt]
            zs = [float(v[3]) for v in vt]
            self.assertEqual(len(vt), 8, "two 3 km-apart quads must not weld together")

            m_lat_a, m_lon_a = geo_transform.metres_per_degree(e["lat"])
            # X-Plane: real pos of a local (x, z) vertex placed at the anchor.
            recon = [(e["lat"] - z / m_lat_a, e["lon"] + x / m_lon_a) for x, z in zip(xs, zs)]
            cenA = (sum(p[0] for p in recon[:4]) / 4, sum(p[1] for p in recon[:4]) / 4)
            cenB = (sum(p[0] for p in recon[4:]) / 4, sum(p[1] for p in recon[4:]) / 4)
            # metres off for each piece
            errA = abs(cenA[0] - plat) * m_lat_a + abs(cenA[1] - lonA) * m_lon_a
            errB = abs(cenB[0] - plat) * m_lat_a + abs(cenB[1] - lonB) * m_lon_a
            self.assertLess(errA, 0.15, f"piece A reconstructed {errA:.2f} m off")
            self.assertLess(errB, 0.15, f"piece B reconstructed {errB:.2f} m off")

    def test_coincident_pieces_in_different_texture_groups_stay_registered(self):
        """A crosswalk's red bed and its white bars (or a stand's black ID
        box, its yellow outline and its "A320" text) are separate MSFS
        materials -> separate textures -> separate merge groups. Placed
        coincident in the source, they must come out of the merge coincident
        too. The bug: each group recentred around its OWN footprint and
        picked its OWN metres/degree, so a group that spans the whole apron
        (white bars everywhere) and one that spans a few stands (red beds)
        landed on different anchors and slid ~1-2 m apart."""
        import geo_transform
        with tempfile.TemporaryDirectory() as td:
            obj_dir = Path(td)
            tile_lat, tile_lon = 47, 19

            def _quad(name, tex, lat, lon, half=0.6):
                ir = mesh_ir.MeshIR(
                    name=name, texture=tex, draped=True, draped_layer_offset=-2,
                    positions=np.array([[-half, 0, -half], [half, 0, -half],
                                        [half, 0, half], [-half, 0, half]], dtype=np.float64),
                    normals=np.array([[0, 1, 0]] * 4, dtype=np.float64),
                    uvs=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64),
                    indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.int64),
                )
                mesh_ir.save(ir, mesh_ir.sidecar_path_for(obj_dir / f"{name}.obj"))
                mesh_ir.write_obj8(ir, obj_dir / f"{name}.obj")
                return {"name": name, "lat": lat, "lon": lon, "hdg": 0.0, "agl": 0.0}

            xw = (47.4400, 19.2600)  # the crosswalk
            entries = [
                _quad("white_here", "../textures/xwalk_white.png", *xw),
                _quad("white_faraway", "../textures/xwalk_white.png", 47.4600, 19.3000),  # widens group A
                _quad("red_here", "../textures/xwalk_red.png", *xw),
                _quad("red_nearby", "../textures/xwalk_red.png", 47.4401, 19.2601),      # group B stays local
            ]
            result, _ = draped_merge.merge_draped_layers_in_tile(obj_dir, tile_lat, tile_lon, entries)
            self.assertEqual(len(result), 2)

            def _texture(e):
                return next(l.split(None, 1)[1].strip()
                            for l in (obj_dir / f"{e['name']}.obj").read_text().splitlines()
                            if l.startswith("TEXTURE "))

            def _recon(e):
                vt = [l.split() for l in (obj_dir / f"{e['name']}.obj").read_text().splitlines()
                      if l.startswith("VT ")]
                m_lat, m_lon = geo_transform.metres_per_degree(e["lat"])
                return [(e["lat"] - float(v[3]) / m_lat, e["lon"] + float(v[1]) / m_lon) for v in vt]

            white = next(e for e in result if "white" in _texture(e))
            red = next(e for e in result if "red" in _texture(e))
            # both merged layers must share one placement anchor
            self.assertAlmostEqual(white["lat"], red["lat"], places=9)
            self.assertAlmostEqual(white["lon"], red["lon"], places=9)

            def _closest(pts):
                return min(pts, key=lambda p: (p[0] - xw[0]) ** 2 + (p[1] - xw[1]) ** 2)
            wc, rc = _closest(_recon(white)), _closest(_recon(red))
            m_lat, m_lon = geo_transform.metres_per_degree(xw[0])
            off_m = ((wc[0] - rc[0]) * m_lat) ** 2 + ((wc[1] - rc[1]) * m_lon) ** 2
            self.assertLess(off_m ** 0.5, 0.01,
                            f"coincident crosswalk corner is {off_m ** 0.5 * 1000:.1f} mm out of "
                            f"register between the white and the red merge group")

    def test_different_texture_object_stays_separate(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            shared_uri = _shared_texture_data_uri()
            other_uri = _shared_texture_data_uri()

            glbA = td / "tileA.glb"
            _make_quad_glb(glbA, "TileA", shared_uri, "SharedAsphalt", 0, 10, -10, 0)
            glbB = td / "tileB.glb"
            _make_quad_glb(glbB, "TileB", shared_uri, "SharedAsphalt", 0, 10, 0, 10)
            glbC = td / "tileC.glb"
            _make_quad_glb(glbC, "TileC", other_uri, "StripeTex", 20, 30, -10, 10)

            stemA = mesh_convert.convert(glbA, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")[0].stem
            stemB = mesh_convert.convert(glbB, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")[0].stem
            stemC = mesh_convert.convert(glbC, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")[0].stem

            entries = [
                _placement_entry(obj_dir, "tileA", stemA, 47.0, 8.0),
                _placement_entry(obj_dir, "tileB", stemB, 47.0, 8.0),
                _placement_entry(obj_dir, "tileC", stemC, 47.0, 8.0),
            ]
            result, _ = draped_merge.merge_draped_layers_in_tile(obj_dir, 47, 8, entries)
            names = [e["name"] for e in result]
            self.assertEqual(len(result), 2)
            # stemC has no same-texture merge partner, so it's never welded
            # into the merged A+B mesh -- but it now DOES take part in the
            # tile-wide draped_layer_offset re-rank (see draped_merge.py's
            # own docstring), which can legitimately rename it (a fresh
            # "<stem>_dlrankN" copy) if that changes its offset -- confirm
            # it passed through as its OWN distinct object either way,
            # rather than requiring the exact original name.
            self.assertTrue(
                any(n == stemC or n.startswith(stemC + "_dlrank") for n in names),
                f"different-texture object must pass through as its own object, got {names}")

    def test_two_merge_groups_in_one_tile_get_distinct_offsets(self):
        """Two SEPARATE same-texture merge groups (a big base-fill pair and
        a small stripe-overlay pair, different textures) landing in one
        tile must come out with a DIFFERENT (layer_group, offset) draw slot
        -- an earlier version just inherited whichever original member's own
        offset happened to be listed first (next(...)), so two merge
        results could (and, in real converted packages, did) collide on the
        identical slot despite a comment right above that code claiming to
        rank them against each other. The base fill must also end up
        underneath the stripe (a lower draped draw band, or -- same band --
        a more negative offset)."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            obj_dir.mkdir()
            tex_dir.mkdir()

            base_uri = _shared_texture_data_uri()
            stripe_uri = _shared_texture_data_uri()

            # Big base-fill pair (100m x 20m each half) -- same texture,
            # adjacent along z=0 so they're genuine merge candidates.
            glbA = td / "baseA.glb"
            _make_quad_glb(glbA, "BaseA", base_uri, "BaseAsphalt", 0, 100, -20, 0)
            glbB = td / "baseB.glb"
            _make_quad_glb(glbB, "BaseB", base_uri, "BaseAsphalt", 0, 100, 0, 20)

            # Small stripe-overlay pair (1m x 1m each half) -- different
            # texture from the base fill, also genuine merge candidates
            # with each other, physically inside the base fill's footprint.
            glbC = td / "stripeA.glb"
            _make_quad_glb(glbC, "StripeA", stripe_uri, "PaintStripe", 40, 41, -1, 0)
            glbD = td / "stripeB.glb"
            _make_quad_glb(glbD, "StripeB", stripe_uri, "PaintStripe", 40, 41, 0, 1)

            stemA = mesh_convert.convert(glbA, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")[0].stem
            stemB = mesh_convert.convert(glbB, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")[0].stem
            stemC = mesh_convert.convert(glbC, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")[0].stem
            stemD = mesh_convert.convert(glbD, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")[0].stem

            entries = [
                _placement_entry(obj_dir, "baseA", stemA, 47.0, 8.0),
                _placement_entry(obj_dir, "baseB", stemB, 47.0, 8.0),
                _placement_entry(obj_dir, "stripeA", stemC, 47.0, 8.0),
                _placement_entry(obj_dir, "stripeB", stemD, 47.0, 8.0),
            ]
            result, _ = draped_merge.merge_draped_layers_in_tile(obj_dir, 47, 8, entries)
            self.assertEqual(len(result), 2, "both same-texture pairs should merge into one object each")

            # merge_draped_layers_in_tile only writes a real .obj for its
            # merge results (no .meshir.pkl sidecar -- nothing downstream
            # re-loads a merged result as IR), so read the draw slot back
            # off the ATTR_layer_group_draped line in the written OBJ8 text.
            def _slot_and_texture(entry):
                text = (obj_dir / f"{entry['name']}.obj").read_text(encoding="utf-8")
                line = next(l for l in text.splitlines() if l.startswith("ATTR_layer_group_draped"))
                _, group, offset = line.split()
                texture_line = next(l for l in text.splitlines() if l.startswith("TEXTURE "))
                band_rank = draped_merge._DRAPED_GROUP_ORDER.index(group)
                return (band_rank, int(offset)), texture_line.split(None, 1)[1].strip()

            slots_and_textures = [_slot_and_texture(e) for e in result]
            slots = [s for s, _ in slots_and_textures]
            self.assertEqual(len(set(slots)), 2,
                             f"the two merge results must get distinct (band, offset) slots, got {slots}")

            # Larger footprint (base fill) draws first/underneath -> lower slot.
            base_slot = next(s for s, t in slots_and_textures if "baseasphalt" in t.lower())
            stripe_slot = next(s for s, t in slots_and_textures if "paintstripe" in t.lower())
            self.assertLess(base_slot, stripe_slot)

    def test_small_sign_ranks_above_large_pavement_across_different_files(self):
        """The confirmed real bug: a small taxi-sign panel and a large
        pavement fill, DIFFERENT textures, each the ONLY draped content in
        its OWN source file, independently rank to the same offset (-5)
        under mesh_convert's own per-file ranking -- an undefined X-Plane
        draw-order collision that, in practice, sometimes drew the
        pavement OVER the sign. Neither ever gets a same-texture merge
        partner (draped_merge's own primary mechanism doesn't apply), so
        this is exactly what the tile-wide single-object re-rank exists to
        fix: the small sign must consistently end up ranked ABOVE (more
        positive offset than) the large pavement, tile-wide, regardless of
        which source file each came from."""
        with tempfile.TemporaryDirectory() as td:
            obj_dir = Path(td)

            pavement_ir = mesh_ir.MeshIR(
                name="pavement", texture="../textures/asphalt.png", draped=True, draped_layer_offset=-5,
                positions=np.array([[0, 0, 0], [50, 0, 0], [50, 0, 50], [0, 0, 50]], dtype=np.float64),
                normals=np.array([[0, 1, 0]] * 4, dtype=np.float64),
                uvs=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64),
                indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.int64),
            )
            mesh_ir.save(pavement_ir, mesh_ir.sidecar_path_for(obj_dir / "pavement.obj"))
            mesh_ir.write_obj8(pavement_ir, obj_dir / "pavement.obj")

            sign_ir = mesh_ir.MeshIR(
                name="sign", texture="../textures/sign.png", draped=True, draped_layer_offset=-5,
                positions=np.array([[20, 0, 20], [21, 0, 20], [21, 0, 21], [20, 0, 21]], dtype=np.float64),
                normals=np.array([[0, 1, 0]] * 4, dtype=np.float64),
                uvs=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64),
                indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.int64),
            )
            mesh_ir.save(sign_ir, mesh_ir.sidecar_path_for(obj_dir / "sign.obj"))
            mesh_ir.write_obj8(sign_ir, obj_dir / "sign.obj")

            entries = [
                {"name": "pavement", "lat": 47.0, "lon": 8.0, "hdg": 0.0, "agl": 0.0},
                {"name": "sign", "lat": 47.0, "lon": 8.0, "hdg": 0.0, "agl": 0.0},
            ]
            result, _ = draped_merge.merge_draped_layers_in_tile(obj_dir, 47, 8, entries)
            self.assertEqual(len(result), 2)

            def _slot_of(prefix):
                entry = next(e for e in result if e["name"].startswith(prefix))
                text = (obj_dir / f"{entry['name']}.obj").read_text(encoding="utf-8")
                line = next(l for l in text.splitlines() if l.startswith("ATTR_layer_group_draped"))
                _, group, offset = line.split()
                return (draped_merge._DRAPED_GROUP_ORDER.index(group), int(offset))

            pavement_slot = _slot_of("pavement")
            sign_slot = _slot_of("sign")
            self.assertLess(pavement_slot, sign_slot,
                             "small sign must rank above (later draw slot than) large pavement, tile-wide")

    def test_family_key_folds_bakes_finishes_and_marking_colour_variants(self):
        """One MSFS material exports as dozens of per-placement alpha/colour
        bakes, plus retint finishes for pavement and colour variants for
        line markings (lines_003 / _003y / _003bl / _004r). Every copy must
        collapse to ONE ranking family so the draw band doesn't overflow its
        11 offset slots -- only the distinguishing NUMBER stays, so lines_003
        and lines_004 remain separate."""
        fk = draped_merge._family_key
        # alpha bake, colour bake, colour+alpha bake -> all one family
        self.assertEqual(fk("shs_decal_dirt_01_albd_a87.png"), fk("shs_decal_dirt_01_albd.png"))
        self.assertEqual(fk("shs_decal_dirt_01_albd_cf43_43_43_a96.png"), fk("shs_decal_dirt_01_albd.png"))
        # base-pavement retint finishes fold
        self.assertEqual(fk("lhbp_gp_concretetiles_001_btint_albd.png"),
                         fk("lhbp_gp_concretetiles_001_albd.png"))
        self.assertEqual(fk("lhbp_gp_asphalt_1_dark_albd.png"), fk("lhbp_gp_asphalt_1_albd.png"))
        self.assertEqual(fk("lhbp_gp_asphalt_1_darkest_albd.png"), fk("lhbp_gp_asphalt_1_albd.png"))
        # marking colour variants (glued AND separated) fold onto the number
        base3 = fk("lhbp_gp_lines_003_albd.png")
        self.assertEqual(fk("lhbp_gp_lines_003y_albd.png"), base3)
        self.assertEqual(fk("lhbp_gp_lines_003bl_albd.png"), base3)
        self.assertEqual(fk("lhbp_gp_lines_003_albd_a76.png"), base3)
        base4 = fk("lhbp_gp_lines_004_albd.png")
        self.assertEqual(fk("lhbp_gp_lines_004r_albd.png"), base4)
        self.assertEqual(fk("lhbp_gp_lines_004_b_albd.png"), base4)
        self.assertEqual(fk("lhbp_gp_lines_002_bl_albd.png"), fk("lhbp_gp_lines_002_albd.png"))
        self.assertEqual(fk("lhbp_gp_lines_001_worn_albd.png"), fk("lhbp_gp_lines_001_albd.png"))
        self.assertEqual(fk("lhbp_gp_lines_001_ex_albd.png"), fk("lhbp_gp_lines_001_albd.png"))
        # ...but different NUMBERS stay separate families
        self.assertNotEqual(base3, base4)
        self.assertNotEqual(base3, fk("lhbp_gp_lines_005_albd.png"))

    def test_rank_band_by_family_folds_bakes_in_a_non_marking_band(self):
        """~50 per-placement bakes of ONE dirt decal, plus retint copies of
        ONE apron -- in the runways/taxiways bands they must each collapse to
        a single ranking family so the band doesn't overflow its 11 slots."""
        areas, families = {}, {}
        for i in range(50):
            k = f"_dirt_{i}"
            areas[k] = 3.0
            families[k] = draped_merge._family_key("shs_decal_dirt_01_albd_a%d.png" % (i + 20))
        for tag in ("", "_btint", "_washed", "_dark"):
            k = f"_apron{tag}"
            areas[k] = 500.0
            families[k] = draped_merge._family_key(f"lhbp_gp_concretetiles_001{tag}_albd.png")
        offsets = draped_merge._rank_band_by_family(areas, families)
        self.assertTrue(all(-5 <= o <= 5 for o in offsets.values()))
        self.assertEqual(len({offsets[f"_dirt_{i}"] for i in range(50)}), 1)
        self.assertEqual(len({offsets[f"_apron{t}"] for t in ("", "_btint", "_washed", "_dark")}), 1)

    def test_marking_colour_role_flags_only_the_coloured_bed(self):
        """The one rule fixed by colour in the markings band: a red/green
        ground BED is BED (forced to the bottom slot); everything else --
        white, yellow, black, untagged -- is OTHER and gets ordered by area."""
        role = draped_merge._marking_colour_role
        BED, OTHER = draped_merge._MARKING_ROLE_BED, draped_merge._MARKING_ROLE_OTHER
        self.assertEqual(role("lhbp_gp_lines_004r_albd.png"), BED)      # glued red
        self.assertEqual(role("lhbp_gp_lines_005_r_albd.png"), BED)     # separated red
        self.assertEqual(role("lhbp_gp_parkingasphalt_4_green_albd.png"), BED)
        self.assertEqual(role("lhbp_gp_lines_003bl_albd.png"), OTHER)   # black -> area
        self.assertEqual(role("lhbp_gp_lines_003y_albd.png"), OTHER)    # yellow -> area
        self.assertEqual(role("lhbp_gp_lines_004_albd.png"), OTHER)     # white -> area
        self.assertEqual(role("lhbp_gp_road_marks_albd.png"), OTHER)

    def test_uv_gate_blocks_glyph_from_welding_onto_the_tiled_line_mesh(self):
        """The confirmed stretch bug: a stand-number glyph quad 0.5m across
        shares its texture with the tiled line network, so a position-only
        weld snapped one glyph corner onto a line vertex whose UV is ~2.7
        away in atlas space and the glyph smeared across the sheet. With
        uv_eps set, coincident position + far-apart UV does NOT weld."""
        # glyph quad at origin, UVs in a tiny atlas cell near (0.05, 0.58)
        gp = np.array([[0, 0, 0], [.5, 0, 0], [.5, 0, .5], [0, 0, .5]], dtype=np.float64)
        gu = np.array([[.03, .58], [.09, .58], [.09, .52], [.03, .52]], dtype=np.float64)
        # a line-network vertex sitting exactly on the glyph's first corner,
        # but its UV is tiled way down the sheet at u=2.73
        lp = np.array([[0, 0, 0], [0, 0, -3], [.2, 0, -3]], dtype=np.float64)
        lu = np.array([[2.73, .47], [2.9, .47], [2.9, .5]], dtype=np.float64)
        P = np.vstack([gp, lp]); U = np.vstack([gu, lu])
        N = np.tile([0.0, 1.0, 0.0], (len(P), 1))
        I = np.array([0, 1, 2, 0, 2, 3, 4, 5, 6], dtype=np.int64)
        # position-only weld: the coincident corners (glyph v0 @ (0,0,0),
        # line v4 @ (0,0,0)) fuse -> 7 - 1 = 6 verts, glyph UV corrupted
        p_only, *_ = draped_merge._weld_vertices(P.copy(), N.copy(), U.copy(), I.copy())
        self.assertEqual(len(p_only), 6)
        # uv-gated: same coincident corners, but |ΔUV| ~2.7 >> uv_eps -> no
        # weld, every vertex kept, glyph UVs intact
        p_uv, _n, u_uv, _i = draped_merge._weld_vertices(
            P.copy(), N.copy(), U.copy(), I.copy(), uv_eps=0.05)
        self.assertEqual(len(p_uv), 7)
        self.assertTrue(np.allclose(np.sort(u_uv[:, 0]), np.sort(U[:, 0])),
                        "no glyph corner got a line-network UV")

    def test_markings_merge_is_uv_gated_and_glyph_sits_above_structure(self):
        """End to end through merge_draped_layers_in_tile: a big black stand
        panel (structure) and small yellow legend text (glyph), each merged
        from >=2 same-texture members, come out with the yellow text ABOVE
        the black panel (glyph tier over structure tier). The rest-tier now
        folds colour variants to one _family_key family, so black-panel and
        yellow-text no longer get adjacent per-texture slots -- the
        bed/structure/glyph split is what orders them."""
        with tempfile.TemporaryDirectory() as td:
            obj_dir = Path(td)

            def _quad(name, tex, u0, x0=0.0, side=2.0):
                ir = mesh_ir.MeshIR(
                    name=name, texture=tex, draped=True, draped_layer_offset=0, alpha_mode="MASK",
                    positions=np.array([[x0, 0, 0], [x0 + side, 0, 0], [x0 + side, 0, side], [x0, 0, side]],
                                       dtype=np.float64),
                    normals=np.tile([0.0, 1.0, 0.0], (4, 1)),
                    uvs=np.array([[u0, 0], [u0 + .1, 0], [u0 + .1, .1], [u0, .1]], dtype=np.float64),
                    indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.int64))
                mesh_ir.save(ir, mesh_ir.sidecar_path_for(obj_dir / f"{name}.obj"))
                mesh_ir.write_obj8(ir, obj_dir / f"{name}.obj")

            # black texture: big PANEL members (u=0, 12 m) plus a small glyph
            # member (u=5) on the panel -- the uv-gated weld keeps the two UV
            # clusters apart, and _component_size_split routes the panels to
            # `structure` and the small member to the `glyph` tier (+5).
            _quad("blackPanelA", "../textures/lhbp_gp_lines_003bl_albd.png", 0.0, x0=0.0, side=12.0)
            _quad("blackPanelB", "../textures/lhbp_gp_lines_003bl_albd.png", 0.0, x0=40.0, side=12.0)
            _quad("blackGlyph", "../textures/lhbp_gp_lines_003bl_albd.png", 5.0, x0=2.0, side=1.0)
            _quad("yellowTextA", "../textures/lhbp_gp_lines_003y_albd.png", 3.0, x0=3.0, side=1.0)
            _quad("yellowTextB", "../textures/lhbp_gp_lines_003y_albd.png", 3.0, x0=3.0, side=1.0)
            entries = [{"name": n, "lat": 47.0, "lon": 8.0, "hdg": 0.0, "agl": 0.0}
                       for n in ("blackPanelA", "blackPanelB", "blackGlyph", "yellowTextA", "yellowTextB")]
            result, _ = draped_merge.merge_draped_layers_in_tile(obj_dir, 47, 8, entries)

            merged = []  # (name, texture_basename, group, offset, us)
            for e in result:
                if not e["name"].startswith("_merged_"):
                    continue
                txt = (obj_dir / f"{e['name']}.obj").read_text(encoding="utf-8")
                tex = next(l for l in txt.splitlines() if l.startswith("TEXTURE ")).split("/")[-1].strip()
                grp, off = next(l.split()[1:3] for l in txt.splitlines()
                                if l.startswith("ATTR_layer_group_draped"))
                us = [float(l.split()[7]) for l in txt.splitlines() if l.startswith("VT ")]
                merged.append((e["name"], tex, grp, int(off), us))

            def _one(tex_frag, want):
                hits = [m for m in merged if tex_frag in m[1]
                        and ((want == "glyph") == (m[3] == 5))]
                self.assertEqual(len(hits), 1,
                                 f"expected exactly one {want} obj for {tex_frag}, got {[h[0] for h in hits]}")
                return hits[0]

            _n, _t, bgrp, boff, bus = _one("lines_003bl", "struct")
            self.assertEqual(bgrp, "markings")
            self.assertTrue(all(u < 1.0 for u in bus), f"panel UVs (u~0) not dragged: {bus}")
            _n, _t, _bgg, bgoff, bgus = _one("lines_003bl", "glyph")
            self.assertEqual(bgoff, 5, "the small black member is a glyph, pinned to +5")
            self.assertTrue(all(u > 4.0 for u in bgus), f"glyph UVs (u~5) not dragged: {bgus}")
            _n, _t, ygrp, yoff, yus = _one("lines_003y", "glyph")
            self.assertEqual((ygrp, yoff), ("markings", 5))
            self.assertLess(boff, yoff, "black panel structure sits UNDER the yellow legend text")
            self.assertTrue(all(2.5 < u < 3.5 for u in yus), f"yellow UVs corrupted: {yus}")

    def test_weld_closes_gap_across_grid_cell_boundary(self):
        """Two vertices ~2mm apart that happen to straddle a rounding
        boundary at the weld epsilon's own grid resolution must still
        weld -- pins the neighbor-cell-aware fix (checking a point's own
        quantized cell AND its 26 neighbors, not just its own cell)."""
        eps = draped_merge._MERGE_WELD_EPS_M
        boundary = eps / 2.0  # exactly between cell 0 and cell 1
        positions = np.array([
            [boundary - 0.001, 0.0, 0.0],
            [boundary + 0.001, 0.0, 0.0],
            [10.0, 0.0, 10.0],
        ], dtype=np.float64)
        normals = np.array([[0, 1, 0]] * 3, dtype=np.float64)
        uvs = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float64)
        # Each point its own degenerate triangle, NOT one triangle spanning
        # all 3 -- a real merge never pre-connects two different source
        # objects' vertices (merge_draped_layers_in_tile offsets each
        # member's own indices before concatenating, see its own v_off
        # bookkeeping), so wiring the two near-boundary points together via
        # a shared triangle edge here would trip _weld_vertices' own
        # already-mesh-connected guard for a reason that can't happen in
        # production -- exactly what made this fixture assert 3 instead of
        # 2 before this fix, for a reason unrelated to the neighbor-cell
        # logic actually under test.
        indices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)

        new_positions, new_normals, new_uvs, new_indices = draped_merge._weld_vertices(
            positions, normals, uvs, indices)
        self.assertEqual(len(new_positions), 2, "the two near-boundary points must weld into one")

    def test_weld_vertices_only_indices_restricts_candidacy(self):
        """Two close-together points must weld when both are in
        only_indices, but must NOT weld when one of the pair is excluded --
        pins the safety property a tolerance-raised subset weld depends on:
        a vertex outside only_indices can never be pulled into a weld no
        matter how close another point is, since it never even gets
        bucketed."""
        eps = 2.0
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],   # 0.5m from point 0 -- within eps
            [10.0, 0.0, 10.0],
        ], dtype=np.float64)
        normals = np.array([[0, 1, 0]] * 3, dtype=np.float64)
        uvs = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float64)
        # Degenerate one-point-each triangles -- see the identical note in
        # test_weld_closes_gap_across_grid_cell_boundary above: a real
        # single shared triangle here would pre-connect points 0 and 1 via
        # the already-mesh-connected guard and mask the only_indices
        # behavior actually under test.
        indices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)

        welded, *_ = draped_merge._weld_vertices(positions, normals, uvs, indices, eps=eps, only_indices=[0, 1])
        self.assertEqual(len(welded), 2, "both candidates present -> should weld")

        unwelded, *_ = draped_merge._weld_vertices(positions, normals, uvs, indices, eps=eps, only_indices=[0])
        self.assertEqual(len(unwelded), 3, "point 1 excluded from only_indices -> must stay separate")

    def test_boundary_edges_flags_naked_edges_only(self):
        """A single quad (2 triangles sharing one diagonal) has 4 outer
        edges used once each (naked) and 1 diagonal used twice (interior,
        not naked)."""
        positions = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], dtype=np.float64)
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int64)
        counts = draped_merge._boundary_edges(indices)
        naked = [e for e, c in counts.items() if c == 1]
        interior = [e for e, c in counts.items() if c == 2]
        self.assertEqual(len(naked), 4)
        self.assertEqual(interior, [(0, 2)])

    def _grid_ir(self, name, x0, x1, z0, z1, nx, nz, tex="../textures/apron.png",
                 skip_cells=(), offset=-4):
        """A regularly triangulated draped grid slab, optionally with some
        interior cells (row, col) omitted to punch holes into it."""
        xs = np.linspace(x0, x1, nx + 1)
        zs = np.linspace(z0, z1, nz + 1)
        pos = np.array([[x, 0.0, z] for z in zs for x in xs], dtype=np.float64)
        skip = set(skip_cells)
        idx = []
        for r in range(nz):
            for c in range(nx):
                if (r, c) in skip:
                    continue
                v = r * (nx + 1) + c
                idx += [v, v + 1, v + nx + 1, v + 1, v + nx + 2, v + nx + 1]
        return mesh_ir.MeshIR(
            name=name, texture=tex, draped=True, draped_layer_offset=offset,
            positions=pos,
            normals=np.tile([0.0, 1.0, 0.0], (len(pos), 1)),
            uvs=np.zeros((len(pos), 2)),
            indices=np.array(idx, dtype=np.int64),
        )

    def _stripe_ir(self, name, x0, x1, tex="../textures/crosswalk.png", z0=0.0, z1=3.0):
        # A crosswalk / dashed-line marking is an alpha-cut atlas (MASK), so
        # it welds at the tight 0.20m tolerance -- the loose 0.50m OPAQUE
        # pavement tolerance would fuse stripes with a normal ~0.5m gap.
        return mesh_ir.MeshIR(
            name=name, texture=tex, draped=True, draped_layer_offset=-2, alpha_mode="MASK",
            positions=np.array([[x0, 0, z0], [x1, 0, z0], [x1, 0, z1], [x0, 0, z1]], dtype=np.float64),
            normals=np.array([[0, 1, 0]] * 4, dtype=np.float64),
            uvs=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64),
            indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.int64),
        )

    def test_disjoint_stripe_layer_is_never_hole_filled(self):
        """A crosswalk / dashed-line layer: many small SEPARATE same-texture
        stripe quads with real gaps between them (the black between a zebra
        crossing's white bars). merge_draped_layers_in_tile still pools them
        into one object, but the hole finder must leave every gap alone --
        each stripe is its own component whose only naked ring IS its outer
        perimeter, so there is no interior hole anywhere. The confirmed real
        symptom of getting this wrong: the crossing came out a solid white
        slab."""
        with tempfile.TemporaryDirectory() as td:
            obj_dir = Path(td)
            # 14 stripes, 0.30m wide, 0.80m pitch -> 0.50m gap between each.
            n_stripes = 14
            for i in range(n_stripes):
                x0 = i * 0.8
                ir = self._stripe_ir(f"stripe_{i}", x0, x0 + 0.3)
                mesh_ir.save(ir, mesh_ir.sidecar_path_for(obj_dir / f"stripe_{i}.obj"))
                mesh_ir.write_obj8(ir, obj_dir / f"stripe_{i}.obj")

            entries = [{"name": f"stripe_{i}", "lat": 47.0, "lon": 8.0, "hdg": 0.0, "agl": 0.0}
                       for i in range(n_stripes)]
            logs = []
            result, _ = draped_merge.merge_draped_layers_in_tile(
                obj_dir, 47, 8, entries, log_callback=lambda msg, level="info": logs.append((level, msg)))

            self.assertEqual(len(result), 1, "same-texture stripes still pool into one merged object")
            merged_text = (obj_dir / f"{result[0]['name']}.obj").read_text(encoding="utf-8")
            vt_lines = [l for l in merged_text.splitlines() if l.startswith("VT ")]
            self.assertEqual(len(vt_lines), 4 * n_stripes,
                             "stripes must stay separate -- 4 verts each, none welded or fanned stripe-to-stripe")
            self.assertFalse(any("hole" in msg for _lvl, msg in logs),
                             "a row of separate stripes has no closed interior ring -- nothing to fill or flag")
            self.assertFalse(any(lvl == "warning" for lvl, _msg in logs))

    def test_blink_rigid_agl_and_library_entries_are_excluded(self):
        with tempfile.TemporaryDirectory() as td:
            obj_dir = Path(td)

            blink_ir = mesh_ir.MeshIR(
                name="blink_sign", texture="../textures/shared.png", draped=True, draped_layer_offset=-2,
                positions=np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], dtype=np.float64),
                normals=np.array([[0, 1, 0]] * 4, dtype=np.float64),
                uvs=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64),
                indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.int64),
            )
            # mesh_convert never writes a sidecar for a light_level/anim
            # builder at all -- simulated here by simply not writing one
            # for "blink_sign", exactly matching that real behavior.

            plain_ir = mesh_ir.MeshIR(
                name="plain_fill", texture="../textures/shared.png", draped=True, draped_layer_offset=-2,
                positions=np.array([[2, 0, 0], [3, 0, 0], [3, 0, 1], [2, 0, 1]], dtype=np.float64),
                normals=np.array([[0, 1, 0]] * 4, dtype=np.float64),
                uvs=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64),
                indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.int64),
            )
            mesh_ir.save(plain_ir, mesh_ir.sidecar_path_for(obj_dir / "plain_fill.obj"))
            mesh_ir.write_obj8(plain_ir, obj_dir / "plain_fill.obj")

            rigid_ir = mesh_ir.MeshIR(name="rigid_building", texture="../textures/shared.png", draped=False)
            mesh_ir.save(rigid_ir, mesh_ir.sidecar_path_for(obj_dir / "rigid_building.obj"))

            entries = [
                {"name": "blink_sign", "lat": 47.0, "lon": 8.0, "hdg": 0.0, "agl": 0.0},
                {"name": "plain_fill", "lat": 47.0, "lon": 8.0, "hdg": 0.0, "agl": 0.0},
                {"name": "rigid_building", "lat": 47.0, "lon": 8.0, "hdg": 0.0, "agl": 0.0},
                {"name": "plain_fill", "lat": 47.0, "lon": 8.0, "hdg": 0.0, "agl": 3.5},
                {"name": None, "library_path": "lib/airport/foo.obj", "lat": 47.0, "lon": 8.0, "hdg": 0.0, "agl": 0.0},
            ]
            result, _ = draped_merge.merge_draped_layers_in_tile(obj_dir, 47, 8, entries)
            names = [e.get("name") for e in result]

            self.assertIn("blink_sign", names, "no-sidecar (blink/animated) entry must pass through untouched")
            self.assertIn("rigid_building", names, "rigid (non-draped) object must never be touched")
            # The AGL-mounted plain_fill (agl=3.5) is routed straight to
            # passthrough before the draped-candidate check even runs, so
            # it's always untouched. The ground one (agl=0.0) IS a draped
            # candidate with no same-texture merge partner, so it now also
            # takes part in the tile-wide re-rank (see draped_merge.py's
            # own docstring) and may come back as a fresh "plain_fill_
            # dlrankN" copy if that changes its offset -- either way it
            # must appear as its own separate object.
            plain_fill_variants = [n for n in names if n == "plain_fill" or (n or "").startswith("plain_fill_dlrank")]
            self.assertEqual(len(plain_fill_variants), 2, f"both plain_fill entries (ground + AGL) pass through -- no merge partner, got {names}")
            self.assertIn("plain_fill", plain_fill_variants, "the AGL-mounted one is never touched by re-ranking")
            self.assertTrue(any(e.get("library_path") for e in result))
            self.assertEqual(len(result), 5)

    def test_markings_band_three_tier_bed_structure_glyph(self):
        """The markings-band draw order is decided by GEOMETRY, not colour:
          -5      a solid coloured BED
          -4..+4  structure (panels / lines / fills), ranked by area
          +5      GLYPH sub-meshes (small compact components) -- always on top
        so black-text-on-yellow AND yellow-text-on-black both come out right,
        and a texture used as a big box in one place and small text in
        another is split into a structure part and a glyph part."""
        with tempfile.TemporaryDirectory() as td:
            obj_dir = Path(td)

            def _quad(x0, x1, z0, z1):
                return (np.array([[x0, 0, z0], [x1, 0, z0], [x1, 0, z1], [x0, 0, z1]], dtype=np.float64),
                        np.array([0, 1, 2, 0, 2, 3], dtype=np.int64))

            def _mk(name, tex, quads):
                pos = np.zeros((0, 3)); idx = np.zeros(0, dtype=np.int64)
                for (p, i) in quads:
                    idx = np.concatenate([idx, i + len(pos)]); pos = np.vstack([pos, p])
                ir = mesh_ir.MeshIR(
                    name=name, texture=tex, draped=True, draped_layer_offset=0, alpha_mode="MASK",
                    positions=pos, normals=np.tile([0.0, 1.0, 0.0], (len(pos), 1)),
                    uvs=np.tile([0.05, 0.05], (len(pos), 1)), indices=idx)
                mesh_ir.save(ir, mesh_ir.sidecar_path_for(obj_dir / f"{name}.obj"))
                mesh_ir.write_obj8(ir, obj_dir / f"{name}.obj")

            big = lambda x: [_quad(x, x + 14, 0, 9)]          # 126 m^2 -> structure
            small = lambda x: [_quad(x, x + 1.2, 0, 1.2)]     # 1.4 m^2 -> glyph
            # red bed (two big members)
            _mk("redA", "../textures/lhbp_gp_lines_004r_albd.png", big(0))
            _mk("redB", "../textures/lhbp_gp_lines_004r_albd.png", big(200))
            # black panel (big) + yellow text (small), separate textures
            _mk("blkA", "../textures/lhbp_gp_lines_003bl_albd.png", big(0))
            _mk("blkB", "../textures/lhbp_gp_lines_003bl_albd.png", big(300))
            _mk("ytxtA", "../textures/lhbp_gp_lines_003y_albd.png", small(1))
            _mk("ytxtB", "../textures/lhbp_gp_lines_003y_albd.png", small(400))
            # yellow pentagon (big) + black text (small), separate textures
            _mk("ypnA", "../textures/lhbp_gp_lines_002_y_albd.png", big(0))
            _mk("ypnB", "../textures/lhbp_gp_lines_002_y_albd.png", big(500))
            _mk("btxtA", "../textures/lhbp_gp_lines_002_bl_albd.png", small(2))
            _mk("btxtB", "../textures/lhbp_gp_lines_002_bl_albd.png", small(600))
            # a texture used BOTH as a big box AND (elsewhere) as small text
            _mk("mixA", "../textures/lhbp_gp_lines_007_albd.png", big(0) + small(50))
            _mk("mixB", "../textures/lhbp_gp_lines_007_albd.png", big(700))

            names = ["redA", "redB", "blkA", "blkB", "ytxtA", "ytxtB",
                     "ypnA", "ypnB", "btxtA", "btxtB", "mixA", "mixB"]
            entries = [{"name": n, "lat": 47.0, "lon": 8.0, "hdg": 0.0, "agl": 0.0} for n in names]
            result, _ = draped_merge.merge_draped_layers_in_tile(obj_dir, 47, 8, entries)

            rows = []  # (texture, is_glyph_stem, offset)
            for e in result:
                if not e["name"].startswith("_merged_tile_"):
                    continue
                txt = (obj_dir / f"{e['name']}.obj").read_text(encoding="utf-8")
                tex = next(l for l in txt.splitlines() if l.startswith("TEXTURE ")).split("/")[-1].strip()
                o = int(next(l.split()[2] for l in txt.splitlines()
                             if l.startswith("ATTR_layer_group_draped")))
                rows.append((tex, e["name"].endswith("_glyph"), o))

            def _off(tex, glyph):
                return next(o for t, g, o in rows if t == tex and g == glyph)

            self.assertEqual(_off("lhbp_gp_lines_004r_albd.png", False), -5, "red bed at the bottom")
            # every glyph sub-mesh is pinned to the top slot
            self.assertEqual(_off("lhbp_gp_lines_003y_albd.png", False), 5)   # all-glyph texture
            self.assertEqual(_off("lhbp_gp_lines_002_bl_albd.png", False), 5)
            self.assertEqual(_off("lhbp_gp_lines_007_albd.png", True), 5)     # split-out glyph part
            # structure parts sit below the glyphs, above the bed
            for tex in ("lhbp_gp_lines_003bl_albd.png", "lhbp_gp_lines_002_y_albd.png",
                        "lhbp_gp_lines_007_albd.png"):
                s = _off(tex, False)
                self.assertTrue(-5 < s < 5, f"{tex} structure at {s}")
            # both the hard cases: text on top of its panel, either colour combo
            self.assertLess(_off("lhbp_gp_lines_003bl_albd.png", False),
                            _off("lhbp_gp_lines_003y_albd.png", False))
            self.assertLess(_off("lhbp_gp_lines_002_y_albd.png", False),
                            _off("lhbp_gp_lines_002_bl_albd.png", False))

    def _two_quads_gap(self, gap):
        # quad A x[0,1], quad B x[1+gap, 2+gap], facing across the gap
        pa = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], dtype=np.float64)
        pb = np.array([[1 + gap, 0, 0], [2 + gap, 0, 0], [2 + gap, 0, 1], [1 + gap, 0, 1]], dtype=np.float64)
        P = np.vstack([pa, pb])
        N = np.tile([0.0, 1.0, 0.0], (8, 1))
        U = np.tile([[0.0, 0.0], [1, 0], [1, 1], [0, 1]], (2, 1))
        I = np.array([0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7], dtype=np.int64)
        return P, N, U, I

    def test_weld_vertices_respects_its_eps(self):
        """_weld_vertices (kept as a general mesh utility; the merge pipeline
        no longer calls it) welds two quads whose facing edge sits within
        eps and leaves a wider gap untouched."""
        P, N, U, I = self._two_quads_gap(0.06)
        w, *_ = draped_merge._weld_vertices(P, N, U, I, eps=0.10)
        self.assertEqual(len(w), 6, "a 6cm gap within eps welds 8 verts down to 6")
        P, N, U, I = self._two_quads_gap(0.15)
        w, *_ = draped_merge._weld_vertices(P, N, U, I, eps=0.10)
        self.assertEqual(len(w), 8, "a 15cm gap beyond eps is left untouched")

    def test_dedupe_faces_drops_weld_collapsed_degenerate_triangles(self):
        """A triangle whose weld pulled two corners onto one point (fewer
        than 3 distinct indices) is removed, not kept as a sliver."""
        P = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], dtype=np.float64)
        N = np.tile([0.0, 1.0, 0.0], (4, 1))
        U = np.tile([0.0, 0.0], (4, 1))
        # tri 0 = (0,1,2) good; tri 1 = (0,2,2) degenerate; tri 2 = (0,2,3) good
        I = np.array([0, 1, 2, 0, 2, 2, 0, 2, 3], dtype=np.int64)
        _p, _n, _u, out_i, removed = draped_merge._dedupe_faces(P, N, U, I)
        self.assertEqual(removed, 1)
        self.assertEqual(len(out_i) // 3, 2)
        for t in range(len(out_i) // 3):
            a, b, c = out_i[3 * t:3 * t + 3]
            self.assertEqual(len({int(a), int(b), int(c)}), 3, "no degenerate triangle survives")

    def test_component_size_split_separates_glyphs_from_structure(self):
        """A big panel component + several tiny character components ->
        (structure, glyph); an all-big mesh -> (mesh, None); an all-tiny
        mesh -> (None, mesh)."""
        def _quad(x0, x1, z0, z1):
            return (np.array([[x0, 0, z0], [x1, 0, z0], [x1, 0, z1], [x0, 0, z1]], dtype=np.float64),
                    np.array([0, 1, 2, 0, 2, 3], dtype=np.int64))

        def _mesh(quads):
            pos = np.zeros((0, 3)); idx = np.zeros(0, dtype=np.int64)
            for p, i in quads:
                idx = np.concatenate([idx, i + len(pos)]); pos = np.vstack([pos, p])
            return pos, np.tile([0.0, 1.0, 0.0], (len(pos), 1)), np.tile([0.0, 0.0], (len(pos), 1)), idx

        panel = _quad(0, 20, 0, 10)                     # 200 m^2
        glyphs = [_quad(30 + k, 30 + k + 0.8, 0, 1.0) for k in range(4)]  # ~0.8 m^2 each
        struct, glyph = draped_merge._component_size_split(*_mesh([panel] + glyphs))
        self.assertIsNotNone(struct); self.assertIsNotNone(glyph)
        # structure keeps the 2 panel tris, glyph keeps the 4*2 char tris
        self.assertEqual(len(struct[3]) // 3, 2)
        self.assertEqual(len(glyph[3]) // 3, 8)

        s2, g2 = draped_merge._component_size_split(*_mesh([panel, _quad(40, 60, 0, 10)]))
        self.assertIsNone(g2, "all-big mesh has no glyph part")
        s3, g3 = draped_merge._component_size_split(*_mesh(glyphs))
        self.assertIsNone(s3, "all-tiny mesh is all glyph")

    def test_triangles_to_polygons_uses_real_latlon_and_matching_uv(self):
        """_triangles_to_polygons is the one function standing between a
        welded draped triangle soup and dsf_compiler.build_dsf's own
        "polygons" format -- pins that it (a) produces one dict per
        TRIANGLE, not per source mesh, (b) converts local (x, z) to real
        (lat, lon) via the exact same geo_transform.local_offset_to_latlon
        every other placement/merge path already uses (not a hand-rolled
        reimplementation), and (c) passes uv through completely unchanged
        as (s, t) -- pol_writer.py's whole design depends on this being a
        literal passthrough, not a remapped/flipped copy."""
        positions = np.array([[0, 0, 0], [10, 0, 0], [10, 0, 10], [0, 0, 10]], dtype=np.float64)
        uvs = np.array([[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]], dtype=np.float64)
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int64)
        base_lat, base_lon, heading = 47.0, 8.0, 30.0

        polys = draped_merge._triangles_to_polygons(
            "polygons/foo.pol", base_lat, base_lon, heading, positions, uvs, indices)

        self.assertEqual(len(polys), 2, "one quad (2 triangles) -> 2 polygon dicts")
        for tri in polys:
            self.assertEqual(tri["pol_path"], "polygons/foo.pol")
            self.assertEqual(len(tri["points"]), 3)

        expected_lat0, expected_lon0 = geo_transform.local_offset_to_latlon(base_lat, base_lon, heading, 0.0, 0.0)
        lon0, lat0, s0, t0 = polys[0]["points"][0]
        self.assertAlmostEqual(lon0, expected_lon0, places=9)
        self.assertAlmostEqual(lat0, expected_lat0, places=9)
        self.assertEqual((s0, t0), (0.1, 0.2), "uv must pass through unchanged as (s, t)")

    def test_merge_with_polygons_produces_no_placement_entry(self):
        """use_polygons=True must route merged draped content into real DSF
        polygon primitives instead of an .obj + OBJECT placement entry --
        the whole point of the .pol conversion (a polygon carries its own
        absolute lon/lat per vertex, so it needs no separate placement at
        all)."""
        pol_writer.clear_cache()
        data_uri = _shared_texture_data_uri()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            obj_dir = td / "objects"
            tex_dir = td / "textures"
            polygons_dir = td / "polygons"
            obj_dir.mkdir()
            tex_dir.mkdir()

            glbA = td / "tileA.glb"
            _make_quad_glb(glbA, "TileA", data_uri, "SharedAsphalt", 0, 10, -10, 0)
            glbB = td / "tileB.glb"
            _make_quad_glb(glbB, "TileB", data_uri, "SharedAsphalt", 0, 10, 0, 10)

            resultA = mesh_convert.convert(glbA, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            resultB = mesh_convert.convert(glbB, obj_dir, tex_dir, tex_dir, "0.0", "0.0", "0.0")
            stemA, stemB = resultA[0].stem, resultB[0].stem

            base_lat, base_lon = 47.0, 8.0
            entries = [
                _placement_entry(obj_dir, "tileA", stemA, base_lat, base_lon),
                _placement_entry(obj_dir, "tileB", stemB, base_lat, base_lon),
            ]
            tile_lat, tile_lon = 47, 8
            result, polygons = draped_merge.merge_draped_layers_in_tile(
                obj_dir, tile_lat, tile_lon, entries, use_polygons=True, polygons_dir=polygons_dir)

            self.assertEqual(result, [], "merged draped content must not get an OBJECT placement entry in polygon mode")
            self.assertEqual(len(polygons), 4, "6 welded vertices from the two quads -> 4 triangles, matching the OBJ8 IDX-count/3 case")
            pol_paths = {p["pol_path"] for p in polygons}
            self.assertEqual(len(pol_paths), 1, "both merged pieces share one texture -> one .pol file")
            pol_path = next(iter(pol_paths))
            self.assertTrue((td / pol_path).exists(), f"the referenced .pol file must actually be written to disk: {pol_path}")

    def test_singles_become_polygons_too_and_keep_footprint_draw_order(self):
        """Single (no same-texture merge partner) draped objects -- most
        real taxi/gate signs -- must ALSO become polygons in polygon mode,
        not stay on the OBJ8 path just because they never got a merge
        partner (this is what unifies the plan's Stage 1 base-fill and
        Stage 2 painted-marking/sign work into one pass, since pol_writer
        always uses explicit-UV mode regardless of content type). Emission
        order must still put the larger footprint (the pavement) before
        the small one (the sign) -- DSF draws same-layer-group/offset
        polygons in CMDS stream order, so this is the only thing keeping
        "small thing on top of large thing" correct until Stage 3 adds
        separate per-category layer-group routing."""
        pol_writer.clear_cache()
        with tempfile.TemporaryDirectory() as td:
            obj_dir = Path(td) / "objects"
            polygons_dir = Path(td) / "polygons"
            obj_dir.mkdir()

            pavement_ir = mesh_ir.MeshIR(
                name="pavement", texture="../textures/asphalt.png", draped=True, draped_layer_offset=-5,
                positions=np.array([[0, 0, 0], [50, 0, 0], [50, 0, 50], [0, 0, 50]], dtype=np.float64),
                normals=np.array([[0, 1, 0]] * 4, dtype=np.float64),
                uvs=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64),
                indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.int64),
            )
            mesh_ir.save(pavement_ir, mesh_ir.sidecar_path_for(obj_dir / "pavement.obj"))
            mesh_ir.write_obj8(pavement_ir, obj_dir / "pavement.obj")

            sign_ir = mesh_ir.MeshIR(
                name="sign", texture="../textures/sign.png", draped=True, draped_layer_offset=-5,
                positions=np.array([[20, 0, 20], [21, 0, 20], [21, 0, 21], [20, 0, 21]], dtype=np.float64),
                normals=np.array([[0, 1, 0]] * 4, dtype=np.float64),
                uvs=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64),
                indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.int64),
            )
            mesh_ir.save(sign_ir, mesh_ir.sidecar_path_for(obj_dir / "sign.obj"))
            mesh_ir.write_obj8(sign_ir, obj_dir / "sign.obj")

            entries = [
                {"name": "pavement", "lat": 47.0, "lon": 8.0, "hdg": 0.0, "agl": 0.0},
                {"name": "sign", "lat": 47.0, "lon": 8.0, "hdg": 0.0, "agl": 0.0},
            ]
            result, polygons = draped_merge.merge_draped_layers_in_tile(
                obj_dir, 47, 8, entries, use_polygons=True, polygons_dir=polygons_dir)

            self.assertEqual(result, [], "single draped objects with no merge partner still become polygons in polygon mode")
            self.assertEqual(len(polygons), 4, "2 triangles each for pavement + sign")
            pol_paths = [p["pol_path"] for p in polygons]
            self.assertEqual(len(set(pol_paths)), 2, "different textures -> different .pol files")
            self.assertIn("asphalt", polygons[0]["pol_path"],
                           "larger footprint (pavement) must be emitted first/underneath")
            self.assertIn("sign", polygons[-1]["pol_path"],
                           "smaller footprint (sign) must be emitted last/on top")


    # ------------------------------------------------------------------
    # Pavement seam weld: _weld_component_gaps (the primary green-stripe fix)
    # ------------------------------------------------------------------
    def _two_tiles_with_gap(self, gap):
        """Two 2x2 m quads, each its own 4 verts (2 disconnected components),
        separated by `gap` metres along x. Returns (pos, nrm, uv, idx)."""
        p = np.array([
            [0, 0, 0], [2, 0, 0], [2, 0, 2], [0, 0, 2],
            [2 + gap, 0, 0], [4 + gap, 0, 0], [4 + gap, 0, 2], [2 + gap, 0, 2],
        ], dtype=np.float64)
        nrm = np.tile([0.0, 1.0, 0.0], (8, 1))
        uv = np.tile([0.0, 0.0], (8, 1))
        i = np.array([0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7], dtype=np.int64)
        return p, nrm, uv, i

    def test_weld_joins_two_disconnected_tiles_along_a_subhalfmetre_slit(self):
        """A 0.30 m slit between two disconnected pavement tiles: the boundary
        verts index-merge, so tile A and tile B become ONE connected mesh (no
        inter-component edge left to crack). Vertex count drops; triangle
        count is unchanged."""
        p, nrm, uv, i = self._two_tiles_with_gap(0.30)
        op, on, ou, oi, n = draped_merge._weld_component_gaps(p, nrm, uv, i, eps=0.5)
        self.assertGreater(n, 0, "boundary verts within eps must merge")
        self.assertEqual(len(oi), len(i), "triangle set unchanged")
        self.assertEqual(len(op), 8 - n)
        # one connected component now
        adj = {}
        for a, b, c in oi.reshape(-1, 3):
            for u, v in ((a, b), (b, c), (c, a)):
                adj.setdefault(int(u), set()).add(int(v))
                adj.setdefault(int(v), set()).add(int(u))
        seen, stack = set(), [0]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(adj.get(x, ()))
        self.assertEqual(len(seen), len(op), "tiles must now be one connected mesh")

    def test_weld_leaves_a_wide_gap_open(self):
        """A 3 m gap is wider than eps -- a real opening, not a slit -- so
        nothing merges."""
        p, nrm, uv, i = self._two_tiles_with_gap(3.0)
        op, on, ou, oi, n = draped_merge._weld_component_gaps(p, nrm, uv, i, eps=0.5)
        self.assertEqual(n, 0)
        np.testing.assert_array_equal(op, p)
        np.testing.assert_array_equal(oi, i)

    def test_weld_never_collapses_a_tile_into_itself(self):
        """A lone 0.30 m quad: its 4 corners are naked and within eps of each
        other, but they are all one component (mesh-connected), so
        _weld_vertices must not union any of them."""
        p = np.array([[0, 0, 0], [0.3, 0, 0], [0.3, 0, 0.3], [0, 0, 0.3]], dtype=np.float64)
        nrm = np.tile([0.0, 1.0, 0.0], (4, 1))
        uv = np.zeros((4, 2))
        i = np.array([0, 1, 2, 0, 2, 3], dtype=np.int64)
        op, on, ou, oi, n = draped_merge._weld_component_gaps(p, nrm, uv, i, eps=0.5)
        self.assertEqual(n, 0)
        np.testing.assert_array_equal(op, p)

    def test_weld_does_not_touch_interior_shared_vertices(self):
        """A connected grid slab: interior verts aren't naked (excluded), and
        its perimeter verts aren't near each other -- nothing welds."""
        p, i = self._grid_soup(0.0, 4.0, 0.0, 2.0, nx=2, nz=1)
        nrm = np.tile([0.0, 1.0, 0.0], (len(p), 1))
        uv = np.zeros((len(p), 2))
        op, on, ou, oi, n = draped_merge._weld_component_gaps(p, nrm, uv, i, eps=0.5)
        self.assertEqual(n, 0)
        np.testing.assert_array_equal(op, p)

    def test_cross_object_seam_snap_makes_a_shared_edge_bit_identical(self):
        """_snap_pavement_seams: two SEPARATE pavement objects whose facing
        edges are 0.08 m apart -> the boundary verts of the higher-index one
        snap onto the lower one's exact position (<= eps), so the shared edge
        becomes bit-identical. No index merged; each part keeps its own idx;
        a vertex moves at most eps; interior verts and same-part pairs are
        never touched."""
        # part A: quad x[0,10] z[0,10]; part B: quad x[10.08,20] z[0,10]
        A_xz = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
        B_xz = np.array([[10.08, 0], [20, 0], [20, 10], [10.08, 10]], dtype=np.float64)
        quad = np.array([0, 1, 2, 0, 2, 3], dtype=np.int64)
        parts = [{"xz": A_xz, "idx": quad.copy()}, {"xz": B_xz, "idx": quad.copy()}]
        A0, B0 = A_xz.copy(), B_xz.copy()
        n = draped_merge._snap_pavement_seams(parts, eps=0.15)
        self.assertGreater(n, 0)
        # A (lower index) must not move; B's x=10.08 verts snap to x=10
        np.testing.assert_array_equal(parts[0]["xz"], A0)
        self.assertTrue(np.any(np.isclose(parts[1]["xz"][:, 0], 10.0)))
        self.assertFalse(np.any(np.isclose(parts[1]["xz"][:, 0], 10.08)))
        # B's far edge (x=20) untouched; total move <= eps
        self.assertTrue(np.all(np.isclose(parts[1]["xz"][[1, 2], 0], 20.0)))
        self.assertLessEqual(np.abs(parts[1]["xz"] - B0).max(), 0.15 + 1e-9)
        self.assertEqual(len(parts[1]["idx"]), 6, "no triangle added/removed")

    def test_cross_object_seam_snap_ignores_a_wide_gap(self):
        """A 0.5 m gap between two objects is wider than eps -> left alone."""
        A = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
        B = np.array([[10.5, 0], [20, 0], [20, 10], [10.5, 10]], dtype=np.float64)
        q = np.array([0, 1, 2, 0, 2, 3], dtype=np.int64)
        parts = [{"xz": A.copy(), "idx": q}, {"xz": B.copy(), "idx": q}]
        n = draped_merge._snap_pavement_seams(parts, eps=0.15)
        self.assertEqual(n, 0)
        np.testing.assert_array_equal(parts[1]["xz"], B)

    # ------------------------------------------------------------------
    # Pavement underlay: morphological-close of the coverage union into one
    # solid opaque concrete sheet at the bottom band (_fill_pavement_gaps)
    # ------------------------------------------------------------------
    def _grid_soup(self, x0, x1, z0, z1, nx=3, nz=3, base=0, disconnect=False):
        """(positions, indices) for a triangulated grid slab. disconnect=True
        emits every quad with its OWN 4 vertices."""
        if not disconnect:
            xs = np.linspace(x0, x1, nx + 1)
            zs = np.linspace(z0, z1, nz + 1)
            pos = np.array([[x, 0.0, z] for z in zs for x in xs], dtype=np.float64)
            idx = []
            for r in range(nz):
                for c in range(nx):
                    v = base + r * (nx + 1) + c
                    idx += [v, v + 1, v + nx + 1, v + 1, v + nx + 2, v + nx + 1]
            return pos, np.array(idx, dtype=np.int64)
        pos, idx = [], []
        dx, dz = (x1 - x0) / nx, (z1 - z0) / nz
        for r in range(nz):
            for c in range(nx):
                bx, bz = x0 + c * dx, z0 + r * dz
                v = base + len(pos)
                pos += [[bx, 0.0, bz], [bx + dx, 0.0, bz], [bx + dx, 0.0, bz + dz], [bx, 0.0, bz + dz]]
                idx += [v, v + 1, v + 2, v, v + 2, v + 3]
        return np.array(pos, dtype=np.float64), np.array(idx, dtype=np.int64)

    def _slab(self, x0, x1, z0, z1, tex="../textures/concrete.png", alpha="OPAQUE"):
        p, i = self._grid_soup(x0, x1, z0, z1, 4, 4)
        return {"texture": tex, "alpha": alpha, "area": (x1 - x0) * (z1 - z0),
                "positions": p, "indices": i}

    def _fill_bbox_cover(self, out, res=None):
        """Rasterise a _fill_pavement_gaps result -> set of covered (x,z)
        cell tuples, for asserting where the fill landed."""
        res = res or draped_merge._PAVEMENT_FILL_RES_M
        cells = set()
        for m in out:
            P = m["positions"]
            for t in m["indices"].reshape(-1, 3):
                xs = P[t, 0]
                zs = P[t, 2]
                for x in np.arange(xs.min(), xs.max(), res):
                    for z in np.arange(zs.min(), zs.max(), res):
                        cells.add((round(x / res), round(z / res)))
        return cells

    def test_underlay_seals_a_hairline_seam_between_two_slabs(self):
        """Two slabs with a 0.5 m gap (well under 2*_PAVEMENT_FILL_CLOSE_M) ->
        the morphological close bridges it and the underlay fills the seam
        with a flat quad; no input vertex is touched."""
        a = self._slab(0.0, 20.0, 0.0, 20.0)
        b = self._slab(20.5, 40.0, 0.0, 20.0, tex="../textures/asphalt.png")
        a0, b0 = a["positions"].copy(), b["positions"].copy()
        out = draped_merge._fill_pavement_gaps([a, b])
        self.assertTrue(out, "the 0.5 m seam must be sealed")
        np.testing.assert_array_equal(a["positions"], a0)
        np.testing.assert_array_equal(b["positions"], b0)
        P = out[0]["positions"]
        self.assertLess(P[:, 0].min(), 20.5)
        self.assertGreater(P[:, 0].max(), 20.0)
        self.assertEqual(P[:, 1].max(), 0.0, "underlay is flat / draped")

    def test_underlay_leaves_a_wide_grass_opening_open(self):
        """Two slabs with a 24 m gap -- a real opening, far wider than
        2*_PAVEMENT_FILL_CLOSE_M. The close cannot bridge it, so the middle
        of the gap stays unpaved."""
        a = self._slab(0.0, 20.0, 0.0, 30.0)
        b = self._slab(44.0, 64.0, 0.0, 30.0)
        out = draped_merge._fill_pavement_gaps([a, b])
        if out:
            xs = out[0]["positions"][:, 0]
            self.assertFalse(np.any((xs > 24.0) & (xs < 40.0)),
                             "the middle of a 24 m opening must NOT be paved over")

    def test_underlay_does_not_extend_far_past_a_straight_pavement_edge(self):
        """A single solid slab: a morphological close of a convex shape barely
        grows a straight edge, so the underlay adds at most a ~close-wide
        fringe (never a big apron-sized grey rim on grass)."""
        a = self._slab(0.0, 40.0, 0.0, 40.0)
        out = draped_merge._fill_pavement_gaps([a, dict(a, texture="../textures/x.png")])
        if out:
            P = out[0]["positions"]
            margin = draped_merge._PAVEMENT_FILL_CLOSE_M + draped_merge._PAVEMENT_FILL_RES_M * 2
            self.assertGreaterEqual(P[:, 0].min(), 0.0 - margin)
            self.assertLessEqual(P[:, 0].max(), 40.0 + margin)
            self.assertLessEqual(P[:, 2].max(), 40.0 + margin)

    def test_fill_texture_prefers_a_textured_base_layer_not_blend_or_solid_colour(self):
        """The fill object is drawn OPAQUE (ATTR_no_blend 0.5), so its texture
        must not come from (a) a BLEND layer -- LHBP's real bug: the biggest
        apron piece is BLEND `asphalt_div_a127` whose albedo alpha is ~0, which
        alpha-tests the whole fill away -- nor (b) a GROUND-band solid-colour
        fill (`blackground`/`whiteground` = 1x1 pure black/white). Here BOTH the
        BLEND slab and the GROUND black slab are bigger (verts AND area) than
        the textured BASE slabs, yet the BASE texture must win."""
        def raw(x0, x1, z0, z1, tex, alpha, group, nx, area=0.0):
            p, i = self._grid_soup(x0, x1, z0, z1, nx, nx)
            return {"texture": tex, "alpha": alpha, "group": group,
                    "area": area, "positions": p, "indices": i}
        # every layer shares the same ~1 m terrain-through slit at x 19.5..20.5
        # (the close bridges it); the BLEND and the GROUND-black layers are the
        # biggest (area + verts) but must not supply the underlay texture
        blend_l = raw(0.0, 19.5, 0.0, 20.0, "../textures/asp_div_a127.png",
                      "BLEND", "taxiways", 12, area=9e5)
        blend_r = raw(20.5, 40.0, 0.0, 20.0, "../textures/asp_div_a127.png",
                      "BLEND", "taxiways", 12, area=9e5)
        black_l = raw(0.0, 19.5, 0.0, 20.0, "../textures/blackground.png",
                      "OPAQUE", "shoulders", 12, area=8e5)
        black_r = raw(20.5, 40.0, 0.0, 20.0, "../textures/blackground.png",
                      "OPAQUE", "shoulders", 12, area=8e5)
        left = raw(0.0, 19.5, 0.0, 20.0, "../textures/asp_w_02.png",
                   "OPAQUE", "taxiways", 3)
        right = raw(20.5, 40.0, 0.0, 20.0, "../textures/asp_w_02.png",
                    "OPAQUE", "taxiways", 3)
        out = draped_merge._fill_pavement_gaps(
            [blend_l, blend_r, black_l, black_r, left, right])
        self.assertTrue(out)
        self.assertEqual(out[0]["texture"], "../textures/asp_w_02.png",
                         "fill must sample a textured BASE-band OPAQUE layer, "
                         "never the BLEND or the solid-colour GROUND one")

    def test_fill_end_to_end_bottom_band_and_ranking_untouched(self):
        """Through merge_draped_layers_in_tile: two different-texture pavement
        singles with a 1.5 m gap produce a _seam_bridge_tile_* object at
        'shoulders -5', double-sided -- and MSFS2XP_BRIDGE_SEAMS=0 yields the
        IDENTICAL non-fill entries with the IDENTICAL ATTR_layer_group_draped
        line on each (the fill perturbs no ranking)."""
        def run(td, on):
            obj_dir = Path(td) / "objects"
            obj_dir.mkdir()
            for nm, x0, x1, tex in (("concrete", 0.0, 20.0, "../textures/concrete_apron.png"),
                                    ("asphalt", 21.5, 45.0, "../textures/asphalt_taxi.png")):
                ir = self._grid_ir(nm, x0, x1, 0.0, 20.0, 6, 6, tex=tex)
                mesh_ir.save(ir, mesh_ir.sidecar_path_for(obj_dir / f"{nm}.obj"))
                mesh_ir.write_obj8(ir, obj_dir / f"{nm}.obj")
            entries = [{"name": "concrete", "lat": 47.0, "lon": 8.0, "hdg": 0.0, "agl": 0.0},
                       {"name": "asphalt", "lat": 47.0, "lon": 8.0, "hdg": 0.0, "agl": 0.0}]
            old = os.environ.get("MSFS2XP_BRIDGE_SEAMS")
            os.environ["MSFS2XP_BRIDGE_SEAMS"] = "1" if on else "0"
            try:
                result, _ = draped_merge.merge_draped_layers_in_tile(obj_dir, 47, 8, entries)
            finally:
                if old is None:
                    os.environ.pop("MSFS2XP_BRIDGE_SEAMS", None)
                else:
                    os.environ["MSFS2XP_BRIDGE_SEAMS"] = old
            lines = {}
            for e in result:
                txt = (obj_dir / f"{e['name']}.obj").read_text(encoding="utf-8")
                lines[e["name"]] = next((l for l in txt.splitlines()
                                         if l.startswith("ATTR_layer_group_draped")), None)
            return result, lines, obj_dir

        with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
            on_result, on_lines, on_dir = run(td1, True)
            off_result, off_lines, _ = run(td2, False)

            fills = [e for e in on_result if e["name"].startswith("_seam_bridge_tile_")]
            self.assertEqual(len(fills), 1, "the 1.5 m gap produced one fill object")
            self.assertFalse([e for e in off_result if e["name"].startswith("_seam_bridge_tile_")],
                             "MSFS2XP_BRIDGE_SEAMS=0 disables the pass")
            on_non_fill = {n: l for n, l in on_lines.items() if not n.startswith("_seam_bridge_tile_")}
            self.assertEqual(on_non_fill, off_lines,
                             "every real slab keeps its exact ATTR_layer_group_draped line")
            btxt = (on_dir / f"{fills[0]['name']}.obj").read_text(encoding="utf-8")
            self.assertIn(f"ATTR_layer_group_draped {draped_merge._DRAPED_GROUP_GROUND} -5", btxt)
            self.assertIn("ATTR_no_cull", btxt)

    def test_markings_weld_is_hairline_and_does_not_fuse_separate_bars(self):
        """The crosswalk fix: many MASK stripe quads 0.15 m apart (a zebra
        crossing) merge into ONE pooled object but are NOT fused -- their
        vertices stay put (4 per bar) and the bars stay separate connected
        components. A continuous panel (quads ~0.02 m apart) DOES get
        connected so _component_size_split still sees it as structure."""
        with tempfile.TemporaryDirectory() as td:
            obj_dir = Path(td)
            n = 10
            for i in range(n):                       # zebra: 0.30 m bars, 0.15 m gaps
                x0 = i * 0.45
                ir = self._stripe_ir(f"bar_{i}", x0, x0 + 0.30)
                mesh_ir.save(ir, mesh_ir.sidecar_path_for(obj_dir / f"bar_{i}.obj"))
                mesh_ir.write_obj8(ir, obj_dir / f"bar_{i}.obj")
            entries = [{"name": f"bar_{i}", "lat": 47.0, "lon": 8.0, "hdg": 0.0, "agl": 0.0}
                       for i in range(n)]
            result, _ = draped_merge.merge_draped_layers_in_tile(obj_dir, 47, 8, entries)
            self.assertEqual(len(result), 1)
            txt = (obj_dir / f"{result[0]['name']}.obj").read_text(encoding="utf-8")
            vt = [l for l in txt.splitlines() if l.startswith("VT ")]
            self.assertEqual(len(vt), 4 * n,
                             "zebra bars must NOT be welded/bridged together -- 4 verts each")
            idxs = _idx_from_obj8(txt)
            comp = draped_merge._vertex_components(len(vt), idxs)
            self.assertEqual(len(set(comp.tolist())), n, "each bar is still its own component")

    def test_markings_are_not_pavement_gap_filled(self):
        """The tile-wide pavement fill only pools GROUND/BASE-band, non-BLEND
        slabs -- a painted-marking layer (LINES band, BLEND) is never in it,
        so a crosswalk's inter-bar gaps can't be closed by it."""
        with tempfile.TemporaryDirectory() as td:
            obj_dir = Path(td) / "objects"
            obj_dir.mkdir()
            for nm, x0, x1, tex in (("line_a", 0.0, 4.0, "../textures/gp_lines_003.png"),
                                    ("line_b", 4.15, 9.0, "../textures/gp_lines_004.png")):
                ir = self._grid_ir(nm, x0, x1, 0.0, 4.0, 4, 4, tex=tex)
                ir.alpha_mode = "BLEND"
                mesh_ir.save(ir, mesh_ir.sidecar_path_for(obj_dir / f"{nm}.obj"))
                mesh_ir.write_obj8(ir, obj_dir / f"{nm}.obj")
            entries = [{"name": "line_a", "lat": 47.0, "lon": 8.0, "hdg": 0.0, "agl": 0.0},
                       {"name": "line_b", "lat": 47.0, "lon": 8.0, "hdg": 0.0, "agl": 0.0}]
            result, _ = draped_merge.merge_draped_layers_in_tile(obj_dir, 47, 8, entries)
            self.assertFalse([e for e in result if e["name"].startswith("_seam_bridge_tile_")],
                             "markings-band layers must never be pavement-gap-filled")


class TestDrapedGroupForTexture(unittest.TestCase):
    """draped_merge._draped_group_for_texture -- CONFIRMED REAL BUG this
    pins: a real MSFS ground-poly texture name that glues "tile" straight
    onto another word with no underscore (ini_GP_GEN_SmallTiles_4m_01)
    matched none of this module's own keyword lists at all, silently
    falling through to the _DRAPED_GROUP_LINES default (the "markings"
    band, meant for painted lines/text/signage) instead of the "taxiways"
    base-pavement band it actually belongs in -- once mesh_convert.
    convert() started draping this near-ground-flat content (see
    mesh_convert.convert's own is_near_ground_flat comment), that put real
    tile/paver/ballast ground detail in direct draw-order competition with
    genuine painted line markings for the same narrow ranking tier,
    reported as pavement pieces flickering/z-fighting against each other
    in-sim."""

    def test_small_tiles_is_base_not_markings(self):
        self.assertEqual(
            draped_merge._draped_group_for_texture("ini_GP_GEN_SmallTiles_4m_01_albd.dds"),
            draped_merge._DRAPED_GROUP_BASE)

    def test_ballast_is_base_not_markings(self):
        self.assertEqual(
            draped_merge._draped_group_for_texture("rail_ballast_01_albd.png"),
            draped_merge._DRAPED_GROUP_BASE)

    def test_tileseam_still_wins_as_wear(self):
        """The new bare "tile" keyword must not shadow the existing, more
        specific "tileseam" (dirt/grout-line) classification -- see the
        guard in _draped_group_for_texture."""
        self.assertEqual(
            draped_merge._draped_group_for_texture("concrete_tileseam_dirt_albd.png"),
            draped_merge._DRAPED_GROUP_WEAR)


if __name__ == "__main__":
    unittest.main()
