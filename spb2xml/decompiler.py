"""
Python port of spb2xml's Decompiler.cs -- see NOTICE.md for attribution.

Usage:
    python3 decompiler.py input.spb output.xml /path/to/propdefs/Common

Requires propdefs.py (propdef XML loader) and textdecode.py (string codec)
in the same folder.
"""
import struct
import sys
import uuid
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import re

from propdefs import load_propdefs, PropertyDef, SetDef
from textdecode import decode as text_decode


class SPBException(Exception):
    pass


class UnknownTag:
    """Placeholder used in Decompiler.tags for a GUID that has no matching
    PropertyDef/SetDef in the loaded propdefs. We deliberately don't raise
    here (in read_tag_data) -- we only raise once/if the tag is actually
    used to parse an element, via UnresolvedPropertyError below."""

    def __init__(self, guid):
        self.guid = guid


class UnknownTypeError(SPBException):
    """Raised by parse_property when a property's type name (whether from
    a resolved TypeDef or the raw type_name string) isn't one we know how
    to read. Caught by parse_element, which routes it through the same
    speculative parse_unresolved() fallback as a missing GUID -- treating
    it as a possible container -- rather than crashing outright."""
    pass


class UnresolvedPropertyError(SPBException):
    """Raised when parse_element hits a tag we can't format: either an
    UnknownTag (GUID missing from every loaded propdef) or a PropertyDef
    whose type never resolved (TypeDef missing from every loaded propdef).

    Caught by parse_set. Note the .spb format only stores an explicit
    byte-length for SetDef blocks (and for TEXT/MLTEXT strings) -- plain
    fixed-size properties (LONG, FLOAT, GUID, PBH, ...) have no length
    prefix of their own. So once we don't know a property's type, we can't
    know how many bytes to skip over *just that property* and still land
    correctly on the next sibling. The finest granularity we can safely
    recover to is the nearest enclosing SetDef block, since that block's
    size *is* known -- so that's what gets dropped, not just the single
    offending property."""
    pass


class Reader:
    """Thin BinaryReader-alike over an in-memory buffer."""

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def read_u16(self):
        v = struct.unpack_from("<H", self.data, self.pos)[0]
        self.pos += 2
        return v

    def read_i32(self):
        v = struct.unpack_from("<i", self.data, self.pos)[0]
        self.pos += 4
        return v

    def read_u32(self):
        v = struct.unpack_from("<I", self.data, self.pos)[0]
        self.pos += 4
        return v

    def read_i64(self):
        v = struct.unpack_from("<q", self.data, self.pos)[0]
        self.pos += 8
        return v

    def read_f32(self):
        v = struct.unpack_from("<f", self.data, self.pos)[0]
        self.pos += 4
        return v

    def read_f64(self):
        v = struct.unpack_from("<d", self.data, self.pos)[0]
        self.pos += 8
        return v

    def read_byte(self):
        v = self.data[self.pos]
        self.pos += 1
        return v

    def read_bytes(self, n):
        v = self.data[self.pos:self.pos + n]
        self.pos += n
        return v


class LLA:
    """Mirrors LLA.cs's long/long/uint/int constructor + D2 ToString."""

    def __init__(self, lat_raw, lon_raw, alt1, alt2):
        self.lat = lat_raw * 90.0 / (10001750.0 * 65536.0 * 65536.0)
        self.lon = lon_raw * 360.0 / (65536.0 ** 4)
        alt0 = alt2 + alt1 / (65536.0 * 65536.0)
        self.alt = alt0 * 3.2808399  # meters -> feet

    def to_d2(self):
        def dm(val, pos_ch, neg_ch):
            ch = pos_ch if val >= 0 else neg_ch
            m1 = int(abs(val))
            minutes = (abs(val) - m1) * 60.0
            return f"{ch}{m1}\u00b0{minutes:.2f}'"
        lat_s = dm(self.lat, "N", "S")
        lon_s = dm(self.lon, "E", "W")
        alt_s = f"{'+' if self.alt >= 0 else '-'}{abs(self.alt):09.2f}"
        return f'{lat_s}",{lon_s}",{alt_s}'


DEC_FMT = "{:.3f}"


class Decompiler:
    def __init__(self, spb_path, bank):
        self.bank = bank
        self.r = Reader(open(spb_path, "rb").read())
        self.tags = []
        self.ntags = 0

    # ---- headers ----
    def read_headers(self):
        sig = self.r.read_u16()
        if sig != 0xEBAC:
            raise SPBException("Invalid file ID")
        ntags = 0
        for i in range(12):
            v = self.r.read_i32()
            if i == 6:
                ntags = v
        self.ntags = ntags

    def read_tag_data(self):
        for i in range(self.ntags - 1):
            guid_bytes = self.r.read_bytes(16)
            g = uuid.UUID(bytes_le=guid_bytes)
            de = self.bank.lookup_element(g)
            if de is None:
                print(f"Warning: unbound property GUID {g} "
                      f"(not defined in any loaded propdef) -- will drop on use")
                de = UnknownTag(g)
            self.tags.append(de)
            self.r.read_i32()  # unknown flag

    # ---- tree walk ----
    def decompile(self):
        self.read_headers()
        self.read_tag_data()
        root = ET.Element("_root_")  # dummy wrapper, unwrapped at the end
        self.parse_element(None, root, bound=len(self.r.data))
        return root

    def parse_element(self, current, node, bound):
        """bound = the byte offset we must not read past: the end of the
        nearest enclosing SetDef (or EOF at the top level). Needed so the
        speculative "unresolved tag is probably length-prefixed" guess in
        parse_unresolved() below has something to validate itself against."""
        tagnum = self.r.read_i32() - 1
        if tagnum == -1:
            return
        if tagnum < 0 or tagnum > self.ntags:
            raise SPBException(f"Invalid tag index {tagnum}")
        tag_type = self.tags[tagnum]
        if isinstance(tag_type, SetDef):
            self.parse_set(current, tag_type, node, bound)
        elif isinstance(tag_type, PropertyDef):
            if tag_type.type is None and tag_type.type_name is None:
                self.parse_unresolved(
                    current, node, bound,
                    label=f"property {tag_type.name!r} has no type at all "
                          f"(no type attribute and no matching TypeDef)",
                    guid=tag_type.id, name=tag_type.name)
                return
            try:
                self.parse_property(current, tag_type, node)
            except UnknownTypeError as e:
                self.parse_unresolved(
                    current, node, bound, label=str(e),
                    guid=tag_type.id, name=tag_type.name)
        elif isinstance(tag_type, UnknownTag):
            self.parse_unresolved(
                current, node, bound,
                label=f"tag GUID {tag_type.guid} is not defined in any loaded propdef",
                guid=tag_type.guid, name=None)
        else:
            raise SPBException(f"Unexpected tag type: {type(tag_type).__name__}")

    def parse_unresolved(self, current, node, bound, label, guid, name):
        """We don't know this tag's format. Only SetDef blocks (and
        TEXT/MLTEXT strings, handled separately) are length-prefixed in
        this format -- so guess it's a Set: read the next i32 as a
        candidate size and validate it doesn't run past `bound`.

        For small candidate sizes, "does it fit inside bound" passes by
        pure chance far too often (almost any small int does), which used
        to send us recursing into pure garbage. So for anything with
        actual content (size >= 4) we also peek the first 4 bytes of that
        content and require it to look like a real tag reference (either
        the end-of-set sentinel 0, or tagnum-1 landing inside our known
        tag table) before committing to the guess at all.

        If that holds, this is (almost always) a genuine SetDef whose
        *own* name/GUID just isn't in the loaded propdefs -- its children
        are normally perfectly ordinary, already-known tags. So instead of
        treating the whole span as an opaque blob, recurse into it exactly
        like parse_set does, just under a synthetic <UnresolvedSet
        guid="..."> wrapper instead of the real element name. Any bytes
        that still can't be parsed inside -- a further nested miss, or any
        other SPBException raised while walking the guessed contents --
        get appended as a RawTail hex blob instead of crashing or
        propagating further.

        If the size guess itself doesn't check out (this was actually a
        fixed-size scalar, not a container), raise so the caller falls
        back to dropping the rest of its own enclosing block -- we have
        no way to recover a precise byte length for scalars whose type we
        don't know."""
        save_pos = self.r.pos
        try:
            size = self.r.read_i32()
        except struct.error:
            size = -1
        end_position = self.r.pos + size
        plausible = size >= 0 and end_position <= bound and end_position <= len(self.r.data)
        if plausible and size >= 4:
            try:
                peek = struct.unpack_from("<i", self.r.data, self.r.pos)[0] - 1
            except struct.error:
                plausible = False
            else:
                plausible = peek == -1 or 0 <= peek < self.ntags
        elif plausible and 0 < size < 4:
            plausible = False  # too small to hold even the end-of-set sentinel
        if not plausible:
            self.r.pos = save_pos
            raise UnresolvedPropertyError(label)

        print(f"Warning: {label}; treating as an unnamed container and "
              f"recursing into its {size} bytes (bytes {self.r.pos}-{end_position})")
        wrapper = ET.SubElement(node, "UnresolvedSet")
        wrapper.set("guid", "{" + str(guid).upper() + "}")
        if name:
            wrapper.set("name", name)

        while self.r.pos < end_position:
            child_start = self.r.pos
            try:
                self.parse_element(current, wrapper, bound=end_position)
            except SPBException as e:
                print(f"Warning: {e}; keeping remaining "
                      f"{end_position - child_start} bytes of this unresolved "
                      f"container as a raw tail instead of dropping them")
                self.r.pos = child_start
                tail = ET.SubElement(wrapper, "RawTail")
                tail.text = self.r.data[self.r.pos:end_position].hex()
                self.r.pos = end_position
                break

    def parse_set(self, current, s, node, bound):
        set_size = self.r.read_i32()
        end_position = self.r.pos + set_size
        if end_position > bound:
            raise SPBException(
                f"Set {s.name!r} claims size {set_size} but that overruns "
                f"its enclosing block (end {end_position} > bound {bound})")

        if current is None or s.parent is not current:
            elem_name = f"{s.parent.name}.{s.name}"
            current = s.parent
        else:
            elem_name = s.name

        set_node = ET.SubElement(node, elem_name)
        while self.r.pos < end_position:
            child_start = self.r.pos
            try:
                self.parse_element(current, set_node, bound=end_position)
            except SPBException as e:
                print(f"Warning: {e}; dropping remainder of <{elem_name}> block "
                      f"(bytes {child_start}-{end_position})")
                self.r.pos = end_position
                break

    def parse_property(self, current, prop, node):
        # Prefer the resolved TypeDef's name, but fall back to the raw
        # type_name string straight off the PropertyDef when no TypeDef
        # was registered for it. Many type names (FLOAT2, FLOAT3, FLOAT4,
        # LONG2, ...) are simple built-ins we already know how to read by
        # name -- they don't actually need a <TypeDef> element to exist
        # anywhere in the loaded propdefs for us to parse them correctly.
        tname = prop.type.name if prop.type is not None else prop.type_name
        if tname is None:
            raise UnknownTypeError(f"Property {prop.name!r} has no type name at all")

        # Generic FLOATn / LONGn vector handling (n floats/ints packed
        # back-to-back, comma-joined) -- covers FLOAT2/FLOAT3/FLOAT4/... and
        # LONG2/LONG3/LONG4/... uniformly instead of hardcoding only the
        # specific arities we happened to see before.
        m = re.fullmatch(r"FLOAT(\d+)", tname)
        if m:
            n = int(m.group(1))
            vals = [self.r.read_f32() for _ in range(n)]
            self.add_prop(current, prop, ",".join(DEC_FMT.format(v) for v in vals), node)
            return
        m = re.fullmatch(r"LONG(\d+)", tname)
        if m:
            n = int(m.group(1))
            vals = [self.r.read_i32() for _ in range(n)]
            self.add_prop(current, prop, ",".join(str(v) for v in vals), node)
            return

        if tname in ("TEXT", "MLTEXT"):
            string_len = self.r.read_i32()
            if string_len <= 0:
                s = ""
            else:
                s = text_decode(self.r.read_bytes(string_len))
            self.add_prop(current, prop, s, node)
        elif tname == "ULONG":
            v = self.r.read_u32()
            self.add_prop(current, prop, str(v), node)
        elif tname == "LONG":
            v = self.r.read_i32()
            self.add_prop(current, prop, str(v), node)
        elif tname == "BOOL":
            v = self.r.read_i32()
            self.add_prop(current, prop, "true" if v == 1 else "false", node)
        elif tname == "FLOAT":
            f = self.r.read_f32()
            self.add_prop(current, prop, DEC_FMT.format(f), node)
        elif tname == "DOUBLE":
            v = self.r.read_f64()
            self.add_prop(current, prop, DEC_FMT.format(v), node)
        elif tname == "BYTE4":
            b = [self.r.read_byte() for _ in range(4)]
            self.add_prop(current, prop, ",".join(str(x) for x in b), node)
        elif tname == "GUID":
            gd = self.r.read_bytes(16)
            g = uuid.UUID(bytes_le=gd)
            self.add_prop(current, prop, "{" + str(g).upper() + "}", node)
        elif tname in ("PBH", "PBH32"):
            p = self.r.read_u32() / (65536.0 * 65536.0) * 360.0
            b = self.r.read_u32() / (65536.0 * 65536.0) * 360.0
            h = self.r.read_u32() / (65536.0 * 65536.0) * 360.0
            self.r.read_i32()  # pad
            self.add_prop(current, prop,
                           f"{DEC_FMT.format(p)},{DEC_FMT.format(b)},{DEC_FMT.format(h)}", node)
        elif tname == "ENUM":
            if prop.enum is None:
                raise UnknownTypeError(f"Property {prop.name!r} is an ENUM with no values loaded")
            idx = self.r.read_i32()
            val = prop.enum[idx]
            self.add_prop(current, prop, val, node)
        elif tname == "LLA":
            lat = self.r.read_i64()
            lon = self.r.read_i64()
            alt1 = self.r.read_u32()
            alt2 = self.r.read_i32()
            lla = LLA(lat, lon, alt1, alt2)
            self.add_prop(current, prop, lla.to_d2(), node)
        elif tname == "FILETIME":
            pass  # TODO, same as upstream (unhandled)
        else:
            raise UnknownTypeError(f"Don't know how to format type {tname!r} at {self.r.pos}")

    def add_prop(self, current, pd, text, node):
        if pd.is_attribute:
            node.set(pd.name, text)
        else:
            prop_name = pd.name
            if pd.symbol_context is not None and pd.symbol_context is not current:
                prop_name = f"{pd.symbol_context.name}.{prop_name}"
            p_node = ET.SubElement(node, prop_name.rstrip())
            p_node.text = text


def prettify(elem):
    rough = ET.tostring(elem, encoding="utf-8")
    return minidom.parseString(rough).toprettyxml(indent="  ")


def decompile_file(spb_path, xml_path, propdefs_dir):
    bank = load_propdefs(propdefs_dir)
    dec = Decompiler(spb_path, bank)
    wrapper = dec.decompile()
    # unwrap: the real root is the single child of our dummy wrapper
    children = list(wrapper)
    if len(children) != 1:
        raise SPBException(f"Expected exactly one root element, got {len(children)}")
    root = children[0]
    xml_text = prettify(root)
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_text)
    return xml_path


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 decompiler.py input.spb output.xml propdefs_dir")
        sys.exit(1)
    spb_path, xml_path, propdefs_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    out = decompile_file(spb_path, xml_path, propdefs_dir)
    print(f"Wrote {out}")
