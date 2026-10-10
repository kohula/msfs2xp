Test build: below-zero geometry as negative AGL height, props on the
building floor as placed, vehicles on paths turn.

## What's new

- **Below-zero geometry**: an object whose geometry reaches below its zero
  point (drain channels, quay walls into the water) is placed lower by
  that depth, as a negative AGL height. The geometry itself is no longer
  modified (the lowered `_dn` copies are gone).
- **Props inside buildings** (seats, desks, people) sit on the building's
  floor as the building is actually placed -- including buildings that
  were terrain-fitted with a warp or a skirt lift -- sampled right under
  each prop.
- **Path animations**: a looping animation that both moves and turns a
  part (the airport bus) now does both; before only the move was kept,
  so the bus slid along its path without turning.

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
