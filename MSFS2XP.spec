# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

# Only the .spb decompiler CODE from spb2xml -- NOT spb2xml/propdefs (or
# propdefs.zip). That folder is Microsoft/Asobo's own MSFS SDK "Propdefs"
# XML data (a real copy's manifest.json reads `"creator": "Asobo Studio"`),
# not something this project can legally redistribute inside a packaged
# app. Point the app's own Settings -> "Propdefs folder" field (or the
# MSFS2XP_PROPDEFS_DIR env var) at your own MSFS install/SDK's copy
# instead; SPB decoding degrades gracefully (a clear warning, nothing
# crashes) without it configured.
datas = [
    ('iconfin.ico', '.'),
    ('spb2xml/decompiler.py', 'spb2xml'),
    ('spb2xml/propdefs.py', 'spb2xml'),
    ('spb2xml/textdecode.py', 'spb2xml'),
    ('spb2xml/textdecode_data.py', 'spb2xml'),
]
binaries = []
hiddenimports = ['uuid', 'xml.etree.ElementTree', 'xml.dom.minidom']
tmp_ret = collect_all('py7zr')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='MSFS2XP',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['iconfin.ico'],
)
