Test build: terrain fit back to 1.1.2, a skirt only on steep ground, and
objects meant to sit below the ground no longer float.

## What's new

- **Terrain fit restored**: 1.1.3's leveling made every building worse;
  the fit is back to how it was in 1.1.2.
- **Skirt only on steep ground**: where the ground under a building is
  steep (over a 3 % gradient), it is lifted level to the highest ground
  and a skirt is added below its walls down to the terrain, instead of the
  building (and its roof) being bent to the slope. Everywhere else the
  1.1.2 fit is unchanged.
- **Below-ground objects sink again**: objects meant to reach below the
  ground -- e.g. a drain tile whose channel sits under the surface, or
  anything MSFS places below its own ground level (where it has water) --
  were standing on the ground in X-Plane, their surface hovering above
  the pavement. The drop is now built into their geometry. The log shows
  "N placement(s) reaching below the ground lowered into the terrain".

Kept from 1.1.3: large flat ground sheets draped onto the terrain, flat
clutter removed across the whole airport, only real buildings carry
objects on their floors, doubled placements merged.

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
