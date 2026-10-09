Test build: working jetways with the airport's own look, fewer exclusion
zones, night lighting, self-moving parts, new options.

## What's new

- **Working jetways (X-Plane 12)**: an airport's own jetways keep their
  MSFS look and now dock to your aircraft. Each rigged MSFS jetway is
  split into its moving parts (rotunda, tunnel, telescoping sections, cab,
  wheel leg) and written as one X-Plane jetway object driven by X-Plane's
  own jetway animation; apt.dat gets a jetway row for each one. Where a
  stand has two or three, the extra ones go to the second door. Switch off
  on the Main page (or `--static-jetways`) to keep them static. Needs the
  package's own airport record; with the XP11 legacy apt.dat they stay
  static.
- **Fewer exclusion zones**: objects close together share one zone, tiny
  props get none of their own, and the package's own exclusion boxes are
  merged in -- thousands of rectangles become a few hundred. Exclusions
  can be switched off completely (Main page, `--no-exclusions`).
- **Lights**: MSFS lights aimed at the ground were pointing at the sky;
  MSFS 2024's newer light type is converted (those lamps were dark);
  helper lights hung beside a lamp post now shine from the post's lamp
  (`--no-pole-lights`); small glowing lamp heads with no light of their
  own get a night glow (`--no-head-glow`); a light made for a lamp
  fixture now sits at its lens instead of across the model from it.
- **Night glow**: an even, dim night texture is left off -- vehicles no
  longer glow grey and tower-cab glass no longer greys the view. Lit
  windows and signs stay lit.
- **Parts that move by themselves** (radar dishes, fans) turn in a loop.
- **Materials**: normal maps use the material's own strength (no more
  blotchy glass); untextured parts use their own colour instead of
  another part's texture, and bare white metal is drawn dark grey.
- **openSAM**: an empty `no_autodgs.txt` is written with the apt.dat, so
  openSAM adds no docking guidance of its own (Advanced page,
  `--opensam-dgs` to skip).
- **Program**: the version is shown at the top right; options are split
  into Main and Advanced (several options were hidden below the card's
  edge before); an optional `.log` file of the run (`--log-file`); the
  max texture size no longer resets to 0 when the window opens.
- Models embedded in the scenery now keep their own XML, so their
  behaviours, self-moving parts and jetway rigs are read too.

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
