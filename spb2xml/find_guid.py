"""
Given a decompiled XML (from decompiler.py) and a GUID, find every element
whose text is that GUID and print the smallest enclosing block (its parent),
which is typically the SimPropAttach/LibraryObject entry containing that
object's Position/PBH/Scale/etc.

Usage:
    python3 find_guid.py output.xml {CB70DC95-7196-124C-B1DB-45D184E8C677}
"""
import sys
import xml.etree.ElementTree as ET


def normalize_guid(s):
    return s.strip("{}").upper()


def find_and_print(xml_path, target_guid):
    target = normalize_guid(target_guid)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # build parent map since ElementTree elements don't know their parent
    parent_map = {c: p for p in root.iter() for c in p}

    matches = []
    for elem in root.iter():
        if elem.text and normalize_guid(elem.text) == target:
            matches.append(elem)
        for attr_val in elem.attrib.values():
            if normalize_guid(attr_val) == target:
                matches.append(elem)

    if not matches:
        print(f"No element found containing GUID {target_guid}")
        return

    for m in matches:
        block = parent_map.get(m, m)
        print(f"--- match: <{m.tag}> {'attr' if m.text is None else 'text'} "
              f"in <{block.tag}> ---")
        print(ET.tostring(block, encoding="unicode"))
        print()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 find_guid.py output.xml {GUID}")
        sys.exit(1)
    find_and_print(sys.argv[1], sys.argv[2])
