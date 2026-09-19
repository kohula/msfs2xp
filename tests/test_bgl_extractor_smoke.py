"""
bgl_extractor.py ported verbatim. Real .bgl/.spb fixtures are proprietary
MSFS binary content, not something this environment can synthesize
faithfully -- so this is a smoke test (import + pure-math/pure-logic
helpers) plus the two real bugs fixed this project's development that
ARE directly testable without real binary content: install-root
auto-resolution (a real Steam install was found to place bulk content in
a sibling HLM_Packages/Official/Steam tree, not under "Packages") and the
SimObject texture-search depth fix (a real character model's textures
live two folder levels above the model file, in a shared library folder).
Full-format parsing correctness continues to rely on manual spot-checking
against a real MSFS package, exactly as before -- called out here rather
than glossed over.
"""
import json
import re
import struct
import sys
import tempfile
import unittest
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

import bgl_extractor


def _build_glb(images):
    """Minimal binary glTF container -- JSON chunk only, no BIN chunk --
    just enough for _read_glb_json()/_copy_simobject_model() to parse."""
    gltf = {"asset": {"version": "2.0"}, "images": images, "buffers": []}
    json_bytes = json.dumps(gltf).encode("utf-8")
    while len(json_bytes) % 4:
        json_bytes += b" "
    header = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(json_bytes))
    chunk_header = struct.pack("<II", len(json_bytes), 0x4E4F534A)
    return header + chunk_header + json_bytes


class TestBglExtractorSmoke(unittest.TestCase):
    def test_module_imports_cleanly(self):
        self.assertTrue(hasattr(bgl_extractor, "extract"))
        self.assertTrue(hasattr(bgl_extractor, "parse_bgl"))


class TestPureMathHelpers(unittest.TestCase):
    def test_decode_lonlat_dword_known_values(self):
        # dword=0 -> lat=90 (north pole encoding), dword=0 -> lon=-180.
        self.assertAlmostEqual(bgl_extractor.decode_lonlat_dword(0, is_lat=True), 90.0, places=6)
        self.assertAlmostEqual(bgl_extractor.decode_lonlat_dword(0, is_lat=False), -180.0, places=6)
        # Half the lat dword range (0..536870912 maps to 90..-90) decodes to lat=0.
        self.assertAlmostEqual(bgl_extractor.decode_lonlat_dword(536870912 // 2, is_lat=True), 0.0, places=3)
        # Half the lon dword range (0..805306368 maps to -180..180) decodes to lon=0.
        self.assertAlmostEqual(bgl_extractor.decode_lonlat_dword(805306368 // 2, is_lat=False), 0.0, places=3)

    def test_placement_height_offset_agl_passthrough(self):
        self.assertEqual(bgl_extractor.placement_height_offset(12.5, is_agl=True, airport_alt=300.0), 12.5)

    def test_placement_height_offset_absolute_subtracts_airport_alt(self):
        self.assertEqual(bgl_extractor.placement_height_offset(312.5, is_agl=False, airport_alt=300.0), 12.5)

    def test_format_guid_matches_dotnet_guid_string_layout(self):
        guid_bytes = bytes.fromhex("efbeadde" + "0000" + "0000" + "00" + "00" + "000000000000")
        formatted = bgl_extractor.format_guid(guid_bytes)
        self.assertEqual(formatted, "deadbeef-0000-0000-0000-000000000000")

    def test_looks_like_title_record_empty_is_false(self):
        self.assertFalse(bgl_extractor.looks_like_title_record([]))


class TestResolveAttachOffset(unittest.TestCase):
    """SimPropAttach OffsetXYZ -> real-world lat/lon. Confirmed real
    regression: a hand-rolled heading rotation here disagreed with
    geo_transform's (the convention every other placement in this project
    uses) on the sign of the north/south term -- a building's separate
    "interior" object (LHBP_B_1_8_1) rendered out to the side of its own
    exterior shell instead of inside it."""

    def test_matches_geo_transform_at_zero_pitch_roll(self):
        # pitch/roll are a no-op for a real ground SceneryObject -- the
        # result must be IDENTICAL to calling geo_transform directly for
        # every offset/heading combination, not just agree by coincidence
        # at heading 0.
        import geo_transform
        for hdg in (0.0, 37.0, 90.0, 180.0, 273.5):
            for x, z in ((10.0, 0.0), (0.0, -10.0), (5.0, -8.0), (-6.0, 4.0)):
                lat, lon, dy = bgl_extractor.resolve_attach_offset(
                    x, 2.0, z, 47.5, 19.25, 0.0, 0.0, hdg)
                exp_lat, exp_lon = geo_transform.local_offset_to_latlon(47.5, 19.25, hdg, x, z)
                self.assertAlmostEqual(lat, exp_lat, places=9, msg=f"hdg={hdg} x={x} z={z}")
                self.assertAlmostEqual(lon, exp_lon, places=9, msg=f"hdg={hdg} x={x} z={z}")
                self.assertEqual(dy, 2.0)

    def test_north_offset_at_heading_zero_increases_latitude(self):
        # local -Z is north at heading 0 (this project's fixed convention).
        lat, lon, _ = bgl_extractor.resolve_attach_offset(0.0, 0.0, -50.0, 47.5, 19.25, 0.0, 0.0, 0.0)
        self.assertGreater(lat, 47.5)
        self.assertAlmostEqual(lon, 19.25, places=6)

    def test_east_offset_at_heading_zero_increases_longitude(self):
        lat, lon, _ = bgl_extractor.resolve_attach_offset(50.0, 0.0, 0.0, 47.5, 19.25, 0.0, 0.0, 0.0)
        self.assertGreater(lon, 19.25)
        self.assertAlmostEqual(lat, 47.5, places=6)


def _spb_guid_hex(guid_str: str) -> str:
    """.NET GUID string -> the 32-hex-no-dash key bgl_extractor keys
    placements/existing_placements under (bytes_le, matching format_guid())."""
    return uuid.UUID(guid_str).bytes_le.hex().lower()


class TestExtractSpbPlacementsHeightChain(unittest.TestCase):
    """extract_spb_placements's height_offset for a "self" SimPropAttach.
    Real binary .spb content can't be synthesized (see module docstring),
    so Decompiler.decompile() is patched to hand back a hand-built
    ElementTree in the exact shape it would otherwise produce -- everything
    downstream of that (the actual code under test: anchor lookup,
    resolve_attach_offset, the height math, the returned dict) runs for
    real.

    CONFIRMED REAL BUG (LHBP's ATC tower): the SimPropContainer this
    attach hangs off was itself placed above ground (anchor "alt"=2.146,
    e.g. mounted on a floor/mezzanine), and the attach's own local
    OffsetXYZ.y is 4.915. height_offset used to be set to dy (4.915) --
    only the offset from the CONTAINER's own contact point, silently
    dropping the container's own 2.146 m above the real terrain contact
    point. The tower's separately-placed interior model (LHBP_B_1_8_1, a
    plain SceneryObject at the same lat/lon) has no such chain and got the
    full correct value (7.06) -- so the shell rendered ~2.1 m lower than
    its own interior fixtures (user: "the interior of the tower is getting
    out from the building"). height_offset must equal alt (anchor_alt +
    dy), matching how placement_height_offset() treats a plain
    SceneryObject's alt -- not just this attach's own local offset."""

    def _patched_decompile(self, container_guid, attach_guid, offset_xyz="0.0,4.915,0.0"):
        root = ET.Element("SimPropContainer")
        guid_el = ET.SubElement(root, "SimBase.GUID")
        guid_el.text = container_guid
        attach = ET.SubElement(root, "SimPropAttach")
        attach.set("DisplayName", "TowerShell")
        mdl = ET.SubElement(attach, "WorldBase.MDLGuid")
        mdl.text = attach_guid
        off = ET.SubElement(attach, "OffsetXYZ")
        off.text = offset_xyz
        orient = ET.SubElement(attach, "Orientation")
        orient.text = "0.0,0.0,0.0"
        return root

    def test_self_attach_height_offset_includes_the_anchors_own_elevation(self):
        spb2xml_dir = Path(bgl_extractor.__file__).resolve().parent / "spb2xml"
        sys.path.insert(0, str(spb2xml_dir))
        import decompiler

        container_guid = "{11111111-2222-3333-4444-555555555555}"
        attach_guid = "{66666666-7777-8888-9999-aaaaaaaaaaaa}"
        container_guid_hex = _spb_guid_hex(container_guid)

        # Pre-seed the propdefs cache so load_propdefs (real MSFS SDK XML,
        # not something this test needs) is never actually called.
        propdefs_dir = tempfile.mkdtemp()
        (Path(propdefs_dir) / "dummy.xml").write_text("<x/>", encoding="utf-8")
        bgl_extractor._PROPDEFS_CACHE[propdefs_dir] = {}

        # The container's OWN placement: mounted 2.146 m above the real
        # terrain contact point -- e.g. on a floor/mezzanine, not the
        # ground. This is what the old code silently dropped.
        existing_placements = [{
            "guid": container_guid_hex, "lat": 47.5, "lon": 19.25,
            "alt": 2.146, "height_offset": 2.146,
            "pitch": 0.0, "roll": 0.0, "hdg": 0.0, "is_agl": True,
        }]

        fake_root = self._patched_decompile(container_guid, attach_guid)
        with mock.patch.object(decompiler.Decompiler, "__init__", return_value=None), \
             mock.patch.object(decompiler.Decompiler, "decompile", return_value=fake_root):
            spb_path = Path(tempfile.mkdtemp()) / "Tower_SimPropContainer.spb"
            spb_path.write_bytes(b"")
            found = bgl_extractor.extract_spb_placements(
                spb_path, airport_lat=47.5, airport_lon=19.25, airport_alt=100.0,
                existing_placements=existing_placements,
                _log=lambda *a, **k: None, propdefs_dir=propdefs_dir)

        self.assertEqual(len(found), 1)
        entry = found[0]
        # dy (local OffsetXYZ.y, pitch/roll are 0) is 4.915; the anchor's
        # own alt is 2.146 -- the correct total is their sum, 7.061.
        self.assertAlmostEqual(entry["alt"], 7.061, places=6)
        self.assertAlmostEqual(entry["height_offset"], 7.061, places=6,
            msg="height_offset must be the FULL chain (anchor_alt + dy), "
                "not just this attach's own local offset (4.915) -- that's "
                "the exact ~2.1 m tower-shell/interior desync bug.")
        self.assertNotAlmostEqual(entry["height_offset"], 4.915, places=3)


def _build_riff_model_container(name: str, glb_bytes: bytes) -> bytes:
    """Minimal RIFF/GLTF container matching extract_riff_model's own
    reader: a GXML sub-chunk carrying name="..." and a GLBD sub-chunk
    whose payload doesn't start with the nested "GLB\\x00" wrapper, so
    it's taken as the raw glb bytes verbatim."""
    gxml_payload = f'<ModelData name="{name}"/>'.encode("utf-8")
    if len(gxml_payload) % 2:
        gxml_payload += b"\x00"
    gxml_chunk = b"GXML" + struct.pack("<I", len(gxml_payload)) + gxml_payload

    glbd_payload = glb_bytes
    glbd_chunk = b"GLBD" + struct.pack("<I", len(glbd_payload)) + glbd_payload
    if len(glbd_payload) % 2:
        glbd_chunk += b"\x00"

    body = gxml_chunk + glbd_chunk
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"GLTF" + body


class TestModelFileStemUniqueness(unittest.TestCase):
    """Confirmed real risk: _decompress_and_write_riff_model used to name
    each extracted model "<sanitized-name>_<first-8-hex-chars-of-guid>.glb"
    -- only 32 bits of real uniqueness. A large conversion run combines a
    package's own ModelData GUIDs with on-demand base-game GUIDs
    (extract_from_install_index), easily thousands of distinct models
    total, where an 8-hex-char prefix collision between two genuinely
    different GUIDs becomes a real (not just theoretical) risk. A
    collision meant one model's file silently got overwritten by another's
    at the same computed path -- every placement referencing the
    overwritten GUID would then load the WRONG geometry: wrong shape,
    wrong footprint, wrong orientation, at an otherwise-correct placement.
    Fixed by using the full 32-hex-char GUID instead of just its first
    segment."""

    def test_guids_sharing_only_their_first_segment_get_distinct_file_stems(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)

            # Same first 4 bytes (8 hex chars -- the OLD truncated key),
            # different remaining 12 bytes, and even the same "name" so
            # `base` is identical too -- the worst case for the old scheme.
            guid_a = bytes.fromhex("11223344" + "aa" * 12)
            guid_b = bytes.fromhex("11223344" + "bb" * 12)

            container_a = _build_riff_model_container("SharedName", b"glTF" + b"AAAA_MODEL_A_CONTENT")
            container_b = _build_riff_model_container("SharedName", b"glTF" + b"BBBB_MODEL_B_CONTENT")

            result_a = bgl_extractor._decompress_and_write_riff_model(container_a, 0, len(container_a), guid_a, out_dir)
            result_b = bgl_extractor._decompress_and_write_riff_model(container_b, 0, len(container_b), guid_b, out_dir)

            self.assertIsNotNone(result_a)
            self.assertIsNotNone(result_b)
            _, file_stem_a, _, out_path_a = result_a
            _, file_stem_b, _, out_path_b = result_b

            self.assertNotEqual(file_stem_a, file_stem_b,
                                 "two different GUIDs sharing only their first 8 hex chars must not collide")
            self.assertNotEqual(out_path_a, out_path_b)

            # Both files must independently survive with THEIR OWN content
            # -- neither one silently overwritten by the other.
            self.assertIn(b"AAAA_MODEL_A_CONTENT", out_path_a.read_bytes())
            self.assertIn(b"BBBB_MODEL_B_CONTENT", out_path_b.read_bytes())


class TestInstallRootResolution(unittest.TestCase):
    """Confirmed against a real Steam MSFS install this session: the true
    top-level folder can hold a "Packages" directory with only a handful of
    small support packages, while the actual bulk content -- including
    generic library objects -- lives in a SIBLING "HLM_Packages/Official/
    Steam" tree instead. GUID indexing only scans downward, so pointing
    install_root at "Packages" alone would never reach it."""

    def _make_fake_install(self, tmp: Path):
        root = tmp / "Microsoft Flight Simulator"
        (root / "Packages" / "fs-base-ui").mkdir(parents=True)
        (root / "HLM_Packages" / "Official" / "Steam" / "fs-base").mkdir(parents=True)
        (root / "FlightSimulator.exe").write_bytes(b"")
        return root

    def test_picking_packages_subfolder_resolves_up_to_true_root(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._make_fake_install(Path(td))
            resolved = bgl_extractor._resolve_real_install_root(root / "Packages")
            self.assertEqual(resolved, root)

    def test_picking_true_root_directly_is_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._make_fake_install(Path(td))
            resolved = bgl_extractor._resolve_real_install_root(root)
            self.assertEqual(resolved, root)

    def test_picking_deep_subfolder_still_resolves_up(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._make_fake_install(Path(td))
            deep = root / "HLM_Packages" / "Official" / "Steam" / "fs-base"
            resolved = bgl_extractor._resolve_real_install_root(deep)
            self.assertEqual(resolved, root)

    def test_no_marker_found_falls_back_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            unrelated = Path(td) / "some_random_folder"
            unrelated.mkdir()
            resolved = bgl_extractor._resolve_real_install_root(unrelated)
            self.assertEqual(resolved, unrelated)


class TestSimObjectTextureSearch(unittest.TestCase):
    """Confirmed against a real MSFS character SimObject this session: its
    texture files live TWO directory levels above the model's own .gltf,
    in a shared "texture" folder covering every model variant under that
    category -- not beside the model file."""

    def test_texture_two_levels_up_is_found_and_copied(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            humans = td / "Humans"
            texture_dir = humans / "texture"
            texture_dir.mkdir(parents=True)
            real_texture = texture_dir / "CH_TEST_ALBEDO.PNG.DDS"
            real_texture.write_bytes(b"FAKE_DDS_BYTES")

            model_dir = humans / "SomeCharacter" / "model.Variant"
            model_dir.mkdir(parents=True)
            gltf_path = model_dir / "SomeCharacter_LOD00.gltf"
            gltf_path.write_text(json.dumps({
                "images": [{"uri": "CH_TEST_ALBEDO.PNG.DDS"}],
                "buffers": [],
            }), encoding="utf-8")

            models_dir = td / "output_models"
            models_dir.mkdir()
            stem = bgl_extractor._copy_simobject_model(gltf_path, models_dir)

            self.assertIsNotNone(stem)
            # Staged filename is disambiguated with a source-path hash
            # prefix now (see _copy_simobject_model's own docstring for
            # why) -- it's no longer just gltf_path.name verbatim, so
            # locate it via the function's own returned stem instead of
            # hardcoding the old bare-filename convention.
            self.assertTrue((models_dir / f"{stem}{gltf_path.suffix}").exists())
            copied_texture = models_dir / "CH_TEST_ALBEDO.PNG.DDS"
            self.assertTrue(copied_texture.exists(), "texture 2 levels up in a shared texture/ folder was not found")
            self.assertEqual(copied_texture.read_bytes(), real_texture.read_bytes())

    def test_same_folder_texture_still_works(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            model_dir = td / "SimpleObject" / "model"
            model_dir.mkdir(parents=True)
            gltf_path = model_dir / "simple.gltf"
            (model_dir / "simple_albedo.png").write_bytes(b"FAKE_PNG")
            gltf_path.write_text(json.dumps({"images": [{"uri": "simple_albedo.png"}], "buffers": []}), encoding="utf-8")

            models_dir = td / "output_models"
            models_dir.mkdir()
            bgl_extractor._copy_simobject_model(gltf_path, models_dir)
            self.assertTrue((models_dir / "simple_albedo.png").exists())

    def test_glb_with_external_uri_texture_two_levels_up_is_found(self):
        """A .glb's "self-contained" guarantee only covers its OWN embedded
        BIN chunk -- its images[] can still reference an external URI
        instead of a bufferView, exactly like .gltf. Confirmed against a
        real airport package: install-sourced GSE/vehicle/character
        SimObjects are shipped this way, sharing one texture library
        across many .glb files. Previously only .gltf was read here, so
        every one of those came out with a placeholder texture."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            category = td / "SimObjects" / "Category"
            texture_dir = category / "texture"
            texture_dir.mkdir(parents=True)
            real_texture = texture_dir / "CONE_ALBEDO.PNG.DDS"
            real_texture.write_bytes(b"FAKE_DDS_BYTES")

            model_dir = category / "model.Cone_Full"
            model_dir.mkdir(parents=True)
            glb_path = model_dir / "Cone_Full.glb"
            glb_path.write_bytes(_build_glb([{"uri": "CONE_ALBEDO.PNG.DDS"}]))

            models_dir = td / "output_models"
            models_dir.mkdir()
            stem = bgl_extractor._copy_simobject_model(glb_path, models_dir)

            self.assertIsNotNone(stem)
            self.assertTrue((models_dir / f"{stem}{glb_path.suffix}").exists())
            copied_texture = models_dir / "CONE_ALBEDO.PNG.DDS"
            self.assertTrue(copied_texture.exists(), "external-URI texture referenced from a .glb was not found")
            self.assertEqual(copied_texture.read_bytes(), real_texture.read_bytes())


class TestSimObjectBehaviorXmlStaging(unittest.TestCase):
    """_iter_simobject_models/discover_simobjects previously read a
    SimObject's own ModelInfo/ModelBehaviors XML only far enough to parse
    its GUID, then threw the file path away -- convert()'s
    parse_time_behavior needs that SAME file staged as a real, on-disk
    "<model_stem_without_LOD>.xml" sibling of the copied model to detect
    blink/business-hours/proximity animation triggers at all. Confirms the
    fix: the xml now survives from discovery through to models_dir under
    exactly the filename convert() looks for."""

    def test_copy_simobject_model_stages_xml_under_lod_stripped_name(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            model_dir = td / "SomeObject" / "model.Variant"
            model_dir.mkdir(parents=True)
            glb_path = model_dir / "Thing_LOD00.glb"
            glb_path.write_bytes(_build_glb([]))
            xml_path = model_dir / "Variant.xml"
            xml_path.write_text("<ModelInfo guid='{x}'/><ModelBehaviors>blink stuff</ModelBehaviors>",
                                 encoding="utf-8")

            models_dir = td / "output_models"
            models_dir.mkdir()
            stem = bgl_extractor._copy_simobject_model(glb_path, models_dir, xml_src=xml_path)

            self.assertIsNotNone(stem)
            # stem is now "<hash>_Thing_LOD00" (source-path hash prefix,
            # see _copy_simobject_model's docstring) -- confirms the LOD
            # suffix survived at the true end despite the prefix, since
            # the xml staging regex is anchored to strip it from THERE.
            self.assertTrue(stem.endswith("_Thing_LOD00"), stem)
            xml_stem = re.sub(r"_LOD[0-9]+$", "", stem)
            staged_xml = models_dir / f"{xml_stem}.xml"
            self.assertTrue(staged_xml.exists(), "expected the LOD-stripped-named xml sibling convert() looks for")
            self.assertIn("blink stuff", staged_xml.read_text(encoding="utf-8"))

    def test_copy_simobject_model_with_no_xml_src_is_unaffected(self):
        """Backward-compatible: xml_src defaults to None, matching every
        pre-existing caller/test in this file that doesn't pass it."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            model_dir = td / "SomeObject" / "model.Variant"
            model_dir.mkdir(parents=True)
            glb_path = model_dir / "Thing_LOD00.glb"
            glb_path.write_bytes(_build_glb([]))

            models_dir = td / "output_models"
            models_dir.mkdir()
            stem = bgl_extractor._copy_simobject_model(glb_path, models_dir)
            self.assertIsNotNone(stem)
            self.assertFalse(list(models_dir.glob("*.xml")))

    def _make_sim_object(self, td: Path, guid: str, model_value="Variant", lod_name="Thing_LOD00.glb"):
        obj_dir = td / f"SimObjects_{model_value}"
        obj_dir.mkdir(parents=True)
        (obj_dir / "sim.cfg").write_text(
            f"[fltsim.0]\ntitle=Test {model_value}\nmodel={model_value}\n", encoding="utf-8")
        model_dir = obj_dir / f"model.{model_value}"
        model_dir.mkdir()
        (model_dir / lod_name).write_bytes(_build_glb([]))
        (model_dir / f"{model_value}.xml").write_text(
            f"<ModelInfo guid=\"{guid}\"><LOD ModelFile=\"{lod_name}\" /></ModelInfo>"
            f"<ModelBehaviors>ZULU TIME, seconds) 0.5 % 0.5 &gt;</ModelBehaviors>",
            encoding="utf-8",
        )
        return obj_dir

    def test_iter_simobject_models_yields_xml_path(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            self._make_sim_object(td, "{11111111-2222-3333-4444-555555555555}")
            results = list(bgl_extractor._iter_simobject_models(td))
            self.assertEqual(len(results), 1)
            guid_hex, title, model_src, xml_path = results[0]
            self.assertEqual(title, "Test Variant")
            self.assertTrue(model_src.name.endswith("Thing_LOD00.glb"))
            self.assertIsNotNone(xml_path)
            self.assertTrue(xml_path.exists())
            self.assertIn("ZULU TIME", xml_path.read_text(encoding="utf-8"))

    def test_discover_simobjects_end_to_end_stages_behavior_xml(self):
        """Full path: discover_simobjects (the real Step-1 entry point)
        must leave models_dir with the model AND a same-stem .xml
        convert()'s parse_time_behavior can actually find and parse."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            self._make_sim_object(td, "{aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee}")

            out_dir = td / "out"
            guid_additions, name_additions = bgl_extractor.discover_simobjects(td, out_dir)

            self.assertEqual(len(guid_additions), 1)
            model_stem = next(iter(guid_additions.values()))
            models_dir = out_dir / "models"
            self.assertTrue((models_dir / f"{model_stem}.glb").exists())

            import re as _re
            expected_xml = models_dir / f"{_re.sub(r'_LOD[0-9]+$', '', model_stem)}.xml"
            self.assertTrue(expected_xml.exists(), "convert()'s parse_time_behavior looks for exactly this path")

            # mesh_convert/__init__.py does "from .convert import convert",
            # which shadows the mesh_convert.convert ATTRIBUTE with the
            # function itself -- importlib reaches the real submodule.
            import importlib
            convert_module = importlib.import_module("mesh_convert.convert")
            behavior = convert_module.parse_time_behavior(expected_xml)
            self.assertEqual(behavior, ("blink", 0.5, 0.5))

    def test_two_different_simobjects_sharing_the_generic_model_filename_dont_collide(self):
        """The confirmed real bug: MSFS's own convention is for every
        SimObject to ship its model as a GENERICALLY named "model.glb"
        inside its own dedicated subfolder -- staging by bare filename
        alone used to collide whenever two DIFFERENT SimObjects (here,
        two different GUIDs/titles, each in their own subfolder) happened
        to share that filename, which is the norm for this format, not an
        edge case. The second one used to be silently skipped (its
        distinct content never even reaches models_dir), yet its GUID
        still got mapped to the FIRST one's stem -- rendering the wrong
        model, shifted by the wrong model's own origin-offset, at the
        second's real placement point. Both must now stage to distinct
        files with distinct, independently-loadable content."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            self._make_sim_object(td, "{11111111-1111-1111-1111-111111111111}",
                                   model_value="First", lod_name="model.glb")
            self._make_sim_object(td, "{22222222-2222-2222-2222-222222222222}",
                                   model_value="Second", lod_name="model.glb")

            out_dir = td / "out"
            guid_additions, name_additions = bgl_extractor.discover_simobjects(td, out_dir)

            self.assertEqual(len(guid_additions), 2, "both SimObjects must be discovered, not one skipped")
            stems = list(guid_additions.values())
            self.assertEqual(len(set(stems)), 2, "each SimObject must get its own distinct staged stem")

            models_dir = out_dir / "models"
            for stem in stems:
                self.assertTrue((models_dir / f"{stem}.glb").exists(),
                                 f"expected a real, distinct staged file for stem {stem!r}")

            titles = sorted(name_additions.values())
            self.assertEqual(titles, ["Test First", "Test Second"])


def _tlv_chunk(type_guid_str, value_bytes):
    """Builds one [16-byte type GUID][4-byte length][value] chunk matching
    TerrainVectorDb's own confirmed self-describing property scheme (see
    scan_terrain_vector_db's docstring)."""
    return bgl_extractor.guid_str_to_bytes(type_guid_str) + struct.pack("<I", len(value_bytes)) + value_bytes


# Arbitrary but fixed type-tag GUIDs standing in for the confirmed-real
# ones (see scan_terrain_vector_db) -- only _MATERIAL_TYPE needs to match
# the module's real constant; the others just need to be distinct,
# consistently-reused "property field" markers so a run of several of
# them satisfies the resync's "N consecutive valid chunks" requirement.
_SCALAR_TYPE_A = "11111111-1111-1111-1111-111111111111"
_SCALAR_TYPE_B = "22222222-2222-2222-2222-222222222222"
_MATERIAL_TYPE = "cd28efd0-d3f0-43b6-9f88-4e4707344a04"


class TestTerrainVectorDbScanning(unittest.TestCase):
    """TerrainVectorDb (BGL section 0x65) is MSFS World Editor's own
    vector-polygon/vegetation database -- confirmed, through direct
    reverse-engineering against a real package (iniBuilds' EGLC), to hold
    custom ground-material polygons referencing a package's own
    MaterialLibs/*/Library.xml by GUID, via a self-describing
    [type][length][value] property scheme. These tests use synthetic
    (not real proprietary binary) fixtures built from that confirmed
    structure -- see scan_terrain_vector_db's own docstring for exactly
    what was and wasn't reverse-engineered."""

    def test_guid_str_to_bytes_roundtrips_through_format_guid(self):
        original = "4669688a-bdb0-45f5-b7c4-274d2dca018d"
        self.assertEqual(bgl_extractor.format_guid(bgl_extractor.guid_str_to_bytes(original)), original)
        # braces + uppercase, as apt.dat/Library.xml/extracted_guids.json all use
        self.assertEqual(
            bgl_extractor.format_guid(bgl_extractor.guid_str_to_bytes("{4669688A-BDB0-45F5-B7C4-274D2DCA018D}")),
            original)

    def test_scan_finds_material_reference_and_skips_leading_false_positive(self):
        """The confirmed real bug: a subsection's own leading header bytes
        (not TLV data at all) can coincidentally look like ONE plausible
        chunk, which used to send the whole scan off into garbage before
        it ever reached the real first record -- silently skipping right
        past genuine material references with no way to recover them. This
        fixture deliberately opens with exactly that: one fake-but-
        plausible chunk immediately followed by unparseable bytes (so it
        satisfies only 1 consecutive success, not the required run),
        THEN a real, validated 8+-chunk property list containing a
        material reference."""
        fake_leading_chunk = _tlv_chunk(_SCALAR_TYPE_A, b"\x01\x02\x03\x04")
        garbage_breaking_the_chain = b"\xff" * 40  # not a valid chunk header

        material_guid = "4669688a-bdb0-45f5-b7c4-274d2dca018d"
        real_property_list = b"".join([
            _tlv_chunk(_SCALAR_TYPE_A, b"\x00\x00\x00\x00"),
            _tlv_chunk(_SCALAR_TYPE_B, b"\x00\x00\x00\x00"),
            _tlv_chunk(_MATERIAL_TYPE, bgl_extractor.guid_str_to_bytes(material_guid)),
            _tlv_chunk(_SCALAR_TYPE_A, b"\x00\x00\x00\x00"),
            _tlv_chunk(_SCALAR_TYPE_B, b"\x00\x00\x00\x00"),
            _tlv_chunk(_SCALAR_TYPE_A, b"\x00\x00\x00\x00"),
            _tlv_chunk(_SCALAR_TYPE_B, b"\x00\x00\x00\x00"),
            _tlv_chunk(_SCALAR_TYPE_A, b"\x00\x00\x00\x00"),
            _tlv_chunk(_SCALAR_TYPE_B, b"\x00\x00\x00\x00"),
        ])
        opaque_geometry_tail = b"\x01\x00\x00\x00\x03\x00\x00\x00" + bytes(range(200)) * 3

        data = fake_leading_chunk + garbage_breaking_the_chain + real_property_list + opaque_geometry_tail

        feature_count, materials = bgl_extractor.scan_terrain_vector_db(data, 0, len(data))
        self.assertEqual(feature_count, 1)
        self.assertEqual({bgl_extractor.format_guid(m) for m in materials}, {material_guid})

    def test_scan_finds_multiple_features_across_opaque_geometry_gaps(self):
        """Two separate polygons' property lists, each with their own
        material reference, separated by an opaque (non-TLV) geometry
        blob -- both must be found, proving the resync correctly resumes
        scanning after skipping past geometry it can't parse."""
        def make_record(material_guid):
            return b"".join([
                _tlv_chunk(_SCALAR_TYPE_A, b"\x00\x00\x00\x00"),
                _tlv_chunk(_SCALAR_TYPE_B, b"\x00\x00\x00\x00"),
                _tlv_chunk(_MATERIAL_TYPE, bgl_extractor.guid_str_to_bytes(material_guid)),
                _tlv_chunk(_SCALAR_TYPE_A, b"\x00\x00\x00\x00"),
                _tlv_chunk(_SCALAR_TYPE_B, b"\x00\x00\x00\x00"),
                _tlv_chunk(_SCALAR_TYPE_A, b"\x00\x00\x00\x00"),
                _tlv_chunk(_SCALAR_TYPE_B, b"\x00\x00\x00\x00"),
                _tlv_chunk(_SCALAR_TYPE_A, b"\x00\x00\x00\x00"),
            ])
        opaque_geometry = b"\x01\x00\x00\x00\x03\x00\x00\x00" + bytes(range(150)) * 2

        guid_a = "4669688a-bdb0-45f5-b7c4-274d2dca018d"
        guid_b = "0c59d7dc-0ed9-4d33-ac55-e4438fe41950"
        data = make_record(guid_a) + opaque_geometry + make_record(guid_b) + opaque_geometry

        feature_count, materials = bgl_extractor.scan_terrain_vector_db(data, 0, len(data))
        self.assertEqual(feature_count, 2)
        self.assertEqual({bgl_extractor.format_guid(m) for m in materials}, {guid_a, guid_b})

    def test_find_material_libraries_parses_library_xml(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            lib_dir = td / "MaterialLibs" / "test-materials"
            lib_dir.mkdir(parents=True)
            (lib_dir / "Library.xml").write_text(
                '<Library Version="1.1.0">\n'
                '\t<Material Version="1.5.0" Name="test-asphalt" '
                'Guid="{4669688A-BDB0-45F5-B7C4-274D2DCA018D}" Context="Ground surface">\n'
                '\t</Material>\n'
                '</Library>\n',
                encoding="utf-8")

            materials = bgl_extractor.find_material_libraries(td)
            self.assertEqual(len(materials), 1)
            guid_bytes = bgl_extractor.guid_str_to_bytes("4669688a-bdb0-45f5-b7c4-274d2dca018d")
            self.assertEqual(materials[guid_bytes], "test-asphalt")


if __name__ == "__main__":
    unittest.main()
