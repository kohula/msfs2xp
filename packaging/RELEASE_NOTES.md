Test build: fixes for objects standing at the wrong height and for the
replacement picker's map.

## What's new

- **Props stay on their building's floor**: seats, people, counters and
  other objects inside a building now take their height from the
  building's floor instead of from the X-Plane ground under each of them.
  Where the terrain under a terminal isn't level they were sunk into the
  floor or floating above it. Objects on the open apron or under an open
  canopy still sit on their own ground. X-Plane's terrain is not changed.
- **SimProp container heights**: objects inside a container placed at an
  absolute (sea-level) altitude got that whole altitude as their height
  above ground; they are now placed relative to the airport elevation.
- **Flatten fallback** (off by default): a new checkbox ("apt.dat: flatten
  the terrain inside the airport", or `--flatten`) levels the terrain
  inside the airport boundary, as MSFS does, for an airport where objects
  still float or sink.
- **Replacement picker map**: the "Where it is" map shows the airport's
  pavement and objects again (it was blank apart from the red dot).
- **Diagnostics**: the conversion log has a "Placement heights:" line
  (objects placed at an absolute altitude, models with parts below their
  ground point) and a line counting objects set on their building's floor
  -- please include them when reporting floating or sunken objects.

## Windows

Download `MSFS2XP.exe` and run it. The cache, scratch space and settings
are kept next to the .exe.

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
