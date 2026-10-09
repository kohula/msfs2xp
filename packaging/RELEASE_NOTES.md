Test build: jetways dock in X-Plane 12, exclusion zones reduced, light
and material fixes, new options.

## What's new

- **Jetways**: rigged MSFS jetway models are converted into X-Plane 12
  jetways that keep the scenery's own model and textures. Each one is
  written as an animated object on X-Plane's jetway datarefs and listed
  in apt.dat (1500/1501 rows), so X-Plane's jetway command docks it.
  Stands with more than one jetway assign the extra ones to door 2.
  Main page tick box, or `--static-jetways` for the old static objects.
  Requires the package's airport record and the XP12 apt.dat.
- **Exclusion zones**: footprints are now merged into one grid with small
  gaps closed, so far fewer rectangles are written. Can be turned off
  (Main page, `--no-exclusions`).
- **Lights**: ASOBO light direction corrected (local +Z); the MSFS 2024
  light extension (ASOBO_advanced_light) is read; lights-only models
  next to a lamp post are attached to its lamp (`--keep-bare-lights` to
  disable); emissive lamps without a light source get a halo
  (`--no-lamp-glow` to disable); synthesized fixture lights are placed
  correctly.
- **Emissive maps** that are near-constant and dim are dropped (they only
  added a grey wash at night).
- **AutoPlay animations** from the model XML are written as looping
  animations.
- **Materials**: normal map strength (glTF `scale`) is applied; untextured
  materials use their own colour, darkened by their metallic factor.
- **openSAM**: `no_autodgs.txt` is written next to apt.dat
  (`--opensam-dgs` to skip).
- **Program**: version shown in the window; options split into Main and
  Advanced (some were hidden before); optional `.log` file
  (`--log-file`); max texture size no longer resets to 0.
- Embedded models now keep their XML, so their behaviours, animations
  and jetway rigs are read.

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
