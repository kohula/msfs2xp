## What's new

- **apt.dat from the package's own airport record** (MSFS 2020 and 2024):
  runways with thresholds, approach lights and PAPI/VASI, helipads,
  frequencies, taxiway signs, windsocks, taxiway lights, the ATC taxi
  network (real edges, runway edges, taxiway names and size classes, hot
  zones, vehicle roads) and ramp starts with heading, type, size class and
  airlines. Your converted draped pavement stays what you see; runways and
  aprons are a transparent hard surface under it (options: real X-Plane
  runways with markings, MSFS painted lines). The stock X-Plane airport only
  lends ATC flows, metadata, the beacon and truck routes.
- **Glass** is drawn at a chosen opacity (default 50%, 100% = solid), is
  drawn after a building's solid parts so it no longer hides them, solid
  facades no longer get windows punched out, and terrain-fitted buildings
  keep their lit windows at night.
- **Lighter scenery**: each model uses the most detailed MSFS level of
  detail that fits a size-based triangle budget, small props fade out with
  distance, textures are held to the size of what they're drawn on (max
  configurable, default 2048) and unused ones are removed.
- **Fixes**: tiles with more than 65,535 objects no longer vanish; DDS
  textures X-Plane can't load are decoded instead of passed through; normal
  maps use X-Plane's NORMAL_METALNESS layout (no more fully glossy
  surfaces); MSFS placement scale is applied; newer 92-byte placement
  records are read correctly; buildings and draped ground no longer drift
  1-2 m apart; a re-run clears the previous run's output first.
- **SimProp containers** (interiors, apron lights, jetways) are read even
  without the MSFS SDK Propdefs folder.
- **Headless converter** in the same build: `MSFS2XP.exe cli --help` /
  `./MSFS2XP-x86_64.AppImage cli --help`.

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
