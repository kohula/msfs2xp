# Third-party attribution

`decompiler.py`, `textdecode.py`, `textdecode_data.py`, and `propdefs.py`
in this folder are a Python port of **spb2xml**, a tool for decompiling
Microsoft Flight Simulator's compiled SimProp (`.spb`) files back to XML.

- Original tool (2007, FSX): **lc0277** — http://lc0277.nerim.net. Its
  README states: *"This software and its source code are in the public
  domain. Permission to use, copy, modify, and distribute this program
  for any purpose is hereby granted, without any conditions or
  restrictions."*
- MSFS 2020/2024 fork (propdef caching, recursive extraction, the
  MSFS-specific text-value decoding this port includes): **leppie** —
  https://github.com/leppie/spb2xml. No separate license file was added
  to that fork; it's presented as a continuation of the same public
  domain original.

This is a from-scratch Python reimplementation of the same decompilation
logic, not a copy of the C# source. `spb2xml/propdefs/` (Microsoft/Asobo's
own MSFS SDK "Propdefs" XML data) is deliberately **not** included here or
in any packaged build — point the app's Settings → "Propdefs folder"
field, or the `MSFS2XP_PROPDEFS_DIR` environment variable, at your own copy
from the MSFS SDK.
