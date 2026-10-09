#!/usr/bin/env bash
# Builds dist/MSFS2XP-x86_64.AppImage.
#
#   packaging/build_appimage.sh            (from the repository root)
#
# Needs: Python 3.10+ with tkinter (Debian/Ubuntu: python3-tk), and network
# access to fetch appimagetool. Build on the OLDEST distro you want to
# support -- an AppImage runs only on systems with at least the build
# machine's glibc (the release workflow uses Ubuntu 22.04).
#
# Layout: PyInstaller's onedir build goes to AppDir/opt/msfs2xp and AppRun
# starts it from there. The AppImage is mounted read-only when it runs,
# which is why the app keeps its cache, scratch space and settings next to
# the .AppImage FILE (or in ~/.cache/msfs2xp) -- see app_paths.py -- and
# needs no --appimage-extract.
#
# appimagetool from AppImage/appimagetool embeds the static type2 runtime,
# so the result needs no libfuse2 on the user's system either.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
PYTHON="${PYTHON:-python3}"
BUILD="$ROOT/build/appimage"
APPDIR="$BUILD/AppDir"
VERSION="${VERSION:-$(git describe --tags --always 2>/dev/null || echo dev)}"

"$PYTHON" -c "import tkinter" || { echo "tkinter missing (install python3-tk)"; exit 1; }

rm -rf "$BUILD"
mkdir -p "$BUILD" "$ROOT/dist"

"$PYTHON" -m PyInstaller --noconfirm --clean --onedir --windowed \
    --name MSFS2XP \
    --distpath "$BUILD/pyinstaller-dist" --workpath "$BUILD/pyinstaller-work" --specpath "$BUILD" \
    --add-data "$ROOT/iconfin.ico:." \
    --add-data "$ROOT/packaging/VERSION:." \
    --add-data "$ROOT/spb2xml/decompiler.py:spb2xml" \
    --add-data "$ROOT/spb2xml/propdefs.py:spb2xml" \
    --add-data "$ROOT/spb2xml/textdecode.py:spb2xml" \
    --add-data "$ROOT/spb2xml/textdecode_data.py:spb2xml" \
    --hidden-import uuid --hidden-import xml.etree.ElementTree --hidden-import xml.dom.minidom \
    --hidden-import cli --hidden-import spb_native \
    --collect-all py7zr \
    "$ROOT/main.py"

mkdir -p "$APPDIR/opt" "$APPDIR/usr/share/applications" "$APPDIR/usr/share/icons/hicolor/256x256/apps"
cp -a "$BUILD/pyinstaller-dist/MSFS2XP" "$APPDIR/opt/msfs2xp"
cp "$ROOT/packaging/msfs2xp.png" "$APPDIR/msfs2xp.png"
cp "$ROOT/packaging/msfs2xp.png" "$APPDIR/usr/share/icons/hicolor/256x256/apps/msfs2xp.png"
cp "$ROOT/packaging/msfs2xp.desktop" "$APPDIR/msfs2xp.desktop"
cp "$ROOT/packaging/msfs2xp.desktop" "$APPDIR/usr/share/applications/msfs2xp.desktop"
cp "$ROOT/packaging/AppRun" "$APPDIR/AppRun"
chmod +x "$APPDIR/AppRun"

TOOL="$BUILD/appimagetool-x86_64.AppImage"
curl -fsSL -o "$TOOL" \
    https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage
chmod +x "$TOOL"

OUT="$ROOT/dist/MSFS2XP-x86_64.AppImage"
# --appimage-extract-and-run: build machines (containers, CI) often have no FUSE.
ARCH=x86_64 VERSION="$VERSION" "$TOOL" --appimage-extract-and-run --no-appstream "$APPDIR" "$OUT"
echo "Built $OUT"
