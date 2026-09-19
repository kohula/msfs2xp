"""
Python port of spb2xml's propdef XML loader (SymbolBank / SymbolDef /
PropertyDef / SetDef / TypeDef / EnumDef) -- see NOTICE.md for attribution.

Loads every *.xml file found recursively under a given folder and builds:
  - guid_map:  GUID -> PropertyDef | SetDef   (what ReadTagData needs)
  - type_map:  lowercase type name -> TypeDef  (what ParseProperty needs)

Note: unlike the original C# tool, this does NOT follow
<SymbolInclude filename="..."/> to pull in specific named files -- it
purely loads every .xml file already present under the search directory.
The two approaches end up loading the same files as long as your propdefs
folder actually contains everything (which load_propdefs requires anyway,
since it errors out if it finds zero .xml files); skipping the by-name
include-chasing avoids the original tool's fragile path-resolution rule
(resolving "filename" against the parent of the *including* file's parent
directory), which breaks for folder layouts that don't exactly mirror the
original SDK structure.
"""
import os
import uuid
import xml.etree.ElementTree as ET


def _attr(node, *names):
    for n in names:
        if n in node.attrib:
            return node.attrib[n]
    return None


def _parse_guid(s):
    # Accept "{XXXX-XXXX-...}" or bare form
    return uuid.UUID(s.strip("{}"))


class EnumDef:
    def __init__(self, node):
        self.values = {}
        for child in node:
            tag = child.tag.split("}")[-1]
            if tag.lower() == "enum":
                idx = _attr(child, "value", "Value", "index", "Index")
                name = _attr(child, "name", "Name")
                if idx is not None and name is not None:
                    self.values[int(idx)] = name

    def __getitem__(self, idx):
        return self.values.get(idx, f"UNKNOWN_ENUM_{idx}")


class DefinitionElement:
    def __init__(self, node):
        self.name = _attr(node, "name", "Name")
        gid = _attr(node, "id", "ID", "Id")
        self.id = _parse_guid(gid) if gid else None
        self.desc = _attr(node, "descr")
        self.symbol_context = None  # set by SymbolDef


class PropertyDef(DefinitionElement):
    kind = "property"

    def __init__(self, node, bank):
        super().__init__(node)
        xml_io = _attr(node, "xml_io")
        self.is_attribute = bool(xml_io and xml_io.lower() == "attribute")
        self.type_name = _attr(node, "type", "Type")
        self.type = None  # resolved lazily by bank.resolve_pending_types(),
        # since the TypeDef for this name may live in a file that hasn't
        # been parsed yet at this point (propdef files reference each other
        # via <SymbolInclude> in no particular guaranteed order).
        self.enum = None
        for child in node:
            tag = child.tag.split("}")[-1]
            if tag == "EnumDef":
                self.enum = EnumDef(child)


class SetDef(DefinitionElement):
    kind = "set"

    def __init__(self, node):
        super().__init__(node)
        self.parent = None  # set to owning SymbolDef


class TypeDef(DefinitionElement):
    def __init__(self, node):
        super().__init__(node)
        self.binding_type = None
        for child in node:
            tag = child.tag.split("}")[-1]
            if tag == "binding":
                self.binding_type = _attr(child, "type")


class SymbolDef(DefinitionElement):
    def __init__(self, node, bank):
        super().__init__(node)
        for child in node:
            tag = child.tag.split("}")[-1]
            if tag == "SymbolInclude":
                # Intentionally ignored: load_propdefs() already recursively
                # loads every *.xml file under the search directory, so
                # every file a SymbolInclude could point at gets parsed
                # anyway. Resolving "filename" against the *right* base
                # directory (the original C# tool used a
                # parent-of-parent-of-this-file rule) is fragile and breaks
                # for layouts that don't exactly match the original SDK
                # folder structure -- simplest and most robust is to not
                # depend on it at all.
                pass
            elif tag == "TypeDefs":
                for typeNode in child:
                    if typeNode.tag.split("}")[-1] == "TypeDef":
                        bank.add_type(TypeDef(typeNode))
            elif tag == "PropertyDefs":
                for propNode in child:
                    if propNode.tag.split("}")[-1] == "PropertyDef":
                        prop = PropertyDef(propNode, bank)
                        prop.symbol_context = self
                        bank.add_property(prop)
            elif tag == "SetDefs":
                for setNode in child:
                    if setNode.tag.split("}")[-1] == "SetDef":
                        s = SetDef(setNode)
                        s.parent = self
                        bank.add_set(s)


class SymbolBank:
    def __init__(self):
        self.guid_map = {}
        self.type_map = {}
        self._loaded_files = set()
        self._pending_type_resolution = []

    def add_symbol_definition_file(self, path):
        path = os.path.normpath(path)
        if path in self._loaded_files:
            return
        self._loaded_files.add(path)
        if not os.path.isfile(path):
            print(f"Warning: propdef file not found: {path}")
            return
        tree = ET.parse(path)
        root = tree.getroot()
        # gather every SymbolDef anywhere in the document (matches
        # doc.GetElementsByTagName("SymbolDef") in the original).
        #
        # NOTE: we deliberately do NOT dedup by SymbolDef *name* here.
        # The real MSFS SDK tree ships duplicate/stale copies of several
        # propdef files under two different folders (e.g. both
        # "Propdefs/1.0/propbase.xml" and "Propdefs/1.0/Common/propbase.xml"
        # declare <SymbolDef name="SimBase">, and the Common/ copy is the
        # complete, current one -- the top-level copy is an older, partial
        # leftover). Since load_propdefs() walks every .xml file with no
        # guaranteed order, name-based dedup meant whichever file happened
        # to be visited first "claimed" the namespace name and the other
        # file's entire TypeDefs/PropertyDefs/SetDefs block was silently
        # discarded -- which is exactly how FLOAT3 (and INPUTFLOAT,
        # INPUTBOOL, INPUTLONG, INPUTVARIANT, etc., all defined only in the
        # Common/ copy) ended up unresolved even though their TypeDefs
        # genuinely exist in the propdefs folder.
        #
        # Instead we process every SymbolDef we find and let it merge into
        # the bank: add_type()/add_property()/add_set() already dedup by
        # type-name/GUID internally (first one wins), so processing the
        # same or a differently-named SymbolDef twice is safe -- we simply
        # get the union of everything actually defined, from every file,
        # regardless of walk order.
        for node in root.iter():
            if node.tag.split("}")[-1] == "SymbolDef":
                SymbolDef(node, self)

    def add_type(self, t):
        key = t.name.lower()
        if key not in self.type_map:
            self.type_map[key] = t

    def add_property(self, p):
        if p.id is not None and p.id not in self.guid_map:
            self.guid_map[p.id] = p
            self._pending_type_resolution.append(p)

    def add_set(self, s):
        if s.id is not None and s.id not in self.guid_map:
            self.guid_map[s.id] = s

    def lookup_type(self, name):
        return self.type_map.get(name.lower())

    def lookup_element(self, guid):
        return self.guid_map.get(guid)

    def resolve_pending_types(self):
        """Call once after all propdef files are loaded: fills in
        PropertyDef.type for every property, now that every TypeDef (from
        every included file) is guaranteed to be registered."""
        unresolved = []
        for p in self._pending_type_resolution:
            if p.type_name:
                p.type = self.lookup_type(p.type_name)
            if p.type is None:
                unresolved.append((p.name, p.type_name))
        if unresolved:
            print(f"Warning: {len(unresolved)} propert(y/ies) have unresolved types "
                  f"(first few: {unresolved[:5]})")


def load_propdefs(search_dir):
    """Recursively loads every *.xml file found anywhere under search_dir
    (all subfolders included) as a propdef definition file. Order doesn't
    matter -- <SymbolInclude> is ignored (see module docstring), and
    duplicate GUIDs/type-names are simply skipped (first one wins, same
    as the original tool), so it's safe to point this at a folder that
    contains more than strictly necessary."""
    bank = SymbolBank()
    xml_files = []
    for dirpath, _dirnames, filenames in os.walk(search_dir):
        for fn in filenames:
            if fn.lower().endswith(".xml"):
                xml_files.append(os.path.join(dirpath, fn))
    if not xml_files:
        raise FileNotFoundError(f"No .xml propdef files found anywhere under {search_dir}")
    print(f"Found {len(xml_files)} .xml file(s) under {search_dir}")
    for full in xml_files:
        try:
            bank.add_symbol_definition_file(full)
        except Exception as e:
            print(f"Warning: cannot parse propdef file {full}: {e}")
    bank.resolve_pending_types()
    return bank
