Test build: SimProp objects in the right place, tighter and more complete
exclusions, no more flat objects hovering over the runways.

## What's new

- **SimProp containers placed correctly**: everything inside a SimProp
  container (seats, people, apron gear, jetways) was mirrored front-to-back
  around the container -- a terminal's seats could end up outside the
  building. Fixed.
- **Exclusion zones**: a rotated building no longer gets an exclusion far
  larger than itself (the footprint is now cut into rectangles after
  rotating it, not before), and the package's own exclusion areas are kept
  instead of being replaced by the per-object footprints, so areas the
  scenery author cleared stay cleared.
- **Runway clutter removed**: flat objects lying on the runways (covers,
  plates and flush fixtures under 0.5 m tall, without lights), which hover
  over X-Plane's runway ground, are removed. Aircraft, vehicles, signs,
  lights and painted markings are kept. Checkbox "Remove flat objects
  lying on the runways" (on by default) or `--keep-runway-objects`.

Also in this test series (1.1.1): props stay on their building's floor,
SimProp container heights, a flatten fallback checkbox (off by default),
the replacement picker's map, and a "Placement heights:" log line.

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
