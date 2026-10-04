Test build: level buildings, nothing hovering over the airport ground, no
doubled buildings.

## What's new

- **Buildings stay level on uneven ground**: a building is no longer bent
  to follow the terrain (which tilted its roof). It is lifted as one piece
  to the highest ground under it, and only the bottom edge of its walls
  reaches down to the terrain, like a foundation. Low flat pieces still
  follow the ground.
- **No more hovering ground sheets**: a large flat ground-cover sheet was
  placed as a solid object and hovered metres above the apron like a
  ceiling; large flat sheets on the airport ground are now draped onto
  the terrain. It was also mistaken for a building floor, lifting the
  signs, barriers and vehicles on it -- only real buildings (2.5 m tall
  or more) carry objects on their floors now.
- **Flat clutter across the whole airport**: small flat objects (covers,
  plates, flush fixtures under 0.5 m, no lights) are removed anywhere
  inside the airport, not just on the runway strip. Checkbox "Flat
  objects on the airport ground" (on by default) or `--keep-flat-objects`.
- **No doubled buildings**: the same object reached twice (a raw
  placement and a SimProp container) is now recognised within 0.5 m /
  2 degrees / 0.3 m, so it is placed once instead of twice a few
  centimetres apart.

Also in this test series: SimProp containers placed correctly and tighter
exclusions (1.1.2), props on their building's floor, a flatten fallback
checkbox and the replacement picker's map (1.1.1).

## Windows

Two downloads, the same program:

- `MSFS2XP-win64.zip` -- unzip it anywhere and run `MSFS2XP\MSFS2XP.exe`.
  Recommended: Windows Defender's machine-learning check sometimes
  blocks the single-file exe as "potentially unwanted" (a false positive
  common to self-extracting Python apps); the folder version starts
  faster too.
- `MSFS2XP.exe` -- one file, as before.

The cache, scratch space and settings are kept next to the .exe.

## Linux

```
chmod +x MSFS2XP-x86_64.AppImage
./MSFS2XP-x86_64.AppImage
```

(Browsers strip the executable bit on download -- `chmod +x`, or right-click
-> Properties -> Permissions -> "Allow executing file as program".)

No more `--appimage-extract`: the program runs straight from the AppImage
and keeps its cache, scratch space and settings in `_cache/`, `_temp/` and
`msfs2xp_config.json` **next to the .AppImage file** (or in
`~/.cache/msfs2xp` if that folder isn't writable; `MSFS2XP_DATA_DIR`
overrides both). It also no longer needs `libfuse2` -- any system with
FUSE (`fusermount` or `fusermount3`) works. Built on Ubuntu 22.04, so it
runs on distributions from that age onward.
