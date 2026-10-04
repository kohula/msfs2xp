Diagnostic test build: same conversion as 1.1.4, plus a report of every
placement's height.

## What's new

- **`msfs2xp_placements.csv`** in the output scenery folder: one row per
  converted object -- model name, title, source, position, MSFS altitude
  and whether it is above ground or above sea level, the airport
  elevation, the resulting height above ground, the converter's re-basing
  lift, the model's height, the terrain-fit outcome, whether it was set
  on a building's floor, and what each part finally became. Open it in a
  spreadsheet and filter by model name or position to see why an object
  floats or sinks.

Everything from 1.1.4 is unchanged: terrain fit as in 1.1.2, a skirt
only on steep ground, below-ground drops built into the geometry.

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
