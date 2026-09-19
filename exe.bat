@echo off
setlocal

REM Builds dist\MSFS2XP.exe as a single-file Windows executable.
REM
REM Beyond the app icon, several things have to be bundled explicitly,
REM because bgl_extractor.py's SPB-placement extraction (ground vehicles/
REM SimObject placements) loads them dynamically at runtime (sys.path +
REM import) instead of as normal top-level imports -- so PyInstaller's
REM static analysis never sees them and silently leaves them out by
REM default:
REM
REM   - spb2xml\decompiler.py, propdefs.py, textdecode.py,
REM     textdecode_data.py
REM                        the .spb decompiler code itself. Without these
REM                        bundled, SPB extraction quietly no-ops in the
REM                        packaged EXE (it works fine run from source,
REM                        since bgl_extractor.py finds the real files
REM                        sitting next to itself on disk) -- PyInstaller
REM                        runs a frozen module from a temp extraction
REM                        folder instead, so they have to be bundled to
REM                        still be found there at the same relative
REM                        location.
REM
REM     Deliberately NOT bundled: spb2xml\propdefs\ (and propdefs.zip).
REM     That folder is Microsoft/Asobo's own MSFS SDK "Propdefs" XML data
REM     (a real copy's manifest.json reads `"creator": "Asobo Studio"`) --
REM     not something this project can legally redistribute inside a
REM     packaged app. Point the app's own Settings -> "Propdefs folder"
REM     field (or the MSFS2XP_PROPDEFS_DIR env var) at your own MSFS
REM     install/SDK's copy instead; SPB decoding degrades gracefully
REM     (a clear warning, nothing crashes) without it configured.
REM   - uuid / xml.etree.ElementTree / xml.dom.minidom
REM                        stdlib modules used only inside spb2xml's own
REM                        files, invisible to PyInstaller's analysis for
REM                        the same reason. Without listing them, the
REM                        frozen EXE won't have them bundled and the
REM                        dynamic import fails at runtime.
REM   - py7zr (+ its compiled deps: pyppmd, pybcj, inflate64, brotli,
REM                        pycryptodomex, multivolumefile, texttable, ...)
REM                        terrain_dem.py imports it inside a try/except, so
REM                        PyInstaller's analysis can miss it or its C
REM                        extensions. Without it bundled, the frozen EXE
REM                        silently loses the large-building terrain-fit
REM                        (every big building stays unwarped). --collect-all
REM                        grabs the submodules + binaries + metadata.

python -m PyInstaller --noconsole --onefile --clean --noconfirm --name MSFS2XP ^
    --icon=iconfin.ico ^
    --add-data "iconfin.ico;." ^
    --add-data "spb2xml\decompiler.py;spb2xml" ^
    --add-data "spb2xml\propdefs.py;spb2xml" ^
    --add-data "spb2xml\textdecode.py;spb2xml" ^
    --add-data "spb2xml\textdecode_data.py;spb2xml" ^
    --hidden-import uuid ^
    --hidden-import xml.etree.ElementTree ^
    --hidden-import xml.dom.minidom ^
    --collect-all py7zr ^
    main.py

endlocal
