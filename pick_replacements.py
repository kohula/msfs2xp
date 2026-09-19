#!/usr/bin/env python3
"""
pick_replacements.py -- browser for choosing X-Plane library replacements
for the objects a conversion couldn't resolve.

Some MSFS placements reference assets that simply aren't in the package
being converted -- ASOBO taxiway/runway edge lights, the PAPI, generic
apron clutter linked only by GUID. The converter (main.py) writes every
such still-unresolved object to

    <package>/unresolved_objects.json

(with a friendly name where it can resolve one, a use-count, and a set of
[lat,lon] sample positions). This tool reads that file, shows each object
on a top-down map of the converted airport, and lets you map it to a real
X-Plane "lib/..." path (scanned live from the configured X-Plane install's
own library.txt EXPORT lines) -- or mark it "skip" to place nothing. It
writes

    <package>/object_replacements.json

which main.py reads on the next run (load_object_replacements /
resolve_library_substitution) and applies BEFORE its built-in keyword
table. Picks persist and can be changed any time by re-running this.

    python pick_replacements.py [PACKAGE_DIR] [--xplane XP_ROOT] [--list]

PACKAGE_DIR   defaults to $MSFS2XP_PKG, else the bundled LHBP test path.
--xplane      X-Plane install root; defaults to $MSFS2XP_XPLANE_ROOT.
--list        print the unresolved objects + candidate count and exit
              (no GUI -- for headless checking).
"""
import argparse
import json
import os
import re
import sys
import threading
from pathlib import Path

# Map preview is optional -- the picker still works fully without it.
try:
    from PIL import Image, ImageTk, ImageDraw  # noqa: F401
    _HAVE_PIL = True
except Exception:
    _HAVE_PIL = False

_COPY_SUFFIX_RE = re.compile(r"\s*\(copy\s*\d*\)\s*$", re.IGNORECASE)
_DEFAULT_PKG = "/work/lhbp_src/sofly-airport-lhbp-budapest"
SKIP = "SKIP"


def normalize_title(title):
    if not title:
        return None
    return _COPY_SUFFIX_RE.sub("", title).strip().lower()


def obj_key(title, guid):
    g = str(guid or "").strip().strip("{}").lower()
    return g or (normalize_title(title) or "?")


# --------------------------------------------------------------------------
# data files
# --------------------------------------------------------------------------
def load_unresolved(pkg: Path):
    """[{key,title,guid,count,positions}], newest conversion's list,
    most-used first. `positions` is a list of [lat,lon] (may be empty on
    an older file that predates the field)."""
    try:
        raw = json.loads((pkg / "unresolved_objects.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    objs = raw.get("objects", raw) if isinstance(raw, dict) else raw
    out = []
    for o in objs if isinstance(objs, list) else []:
        if not isinstance(o, dict):
            continue
        title = o.get("title") or ""
        guid = o.get("guid") or ""
        pos = [p for p in (o.get("positions") or [])
               if isinstance(p, (list, tuple)) and len(p) == 2]
        out.append({
            "key": o.get("key") or obj_key(title, guid),
            "title": title, "guid": guid,
            "count": int(o.get("count") or 0),
            "positions": pos,
        })
    out.sort(key=lambda r: (-r["count"], (r["title"] or r["key"]).lower()))
    return out


def load_map_base(pkg: Path):
    """The converter's precomputed map base for this package:
    {"bounds":[la0,la1,lo0,lo1], "pavement_pts":[[lat,lon],...],
     "context":[[lat,lon],...]}  -- a thinned pavement point cloud + a
     marker per placed object, in world coords, so the picker draws a
    top-down airport map without re-parsing the DSF. None if the file
    isn't there (older conversion)."""
    try:
        raw = json.loads((pkg / "unresolved_map.json").read_text(encoding="utf-8"))
        if isinstance(raw, dict) and raw.get("bounds"):
            return raw
    except (OSError, ValueError):
        pass
    return None


def load_existing(pkg: Path):
    """{key: 'lib/...' | 'SKIP'} already chosen for this package."""
    try:
        raw = json.loads((pkg / "object_replacements.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    entries = raw.get("replacements", raw) if isinstance(raw, dict) else {}
    out = {}
    if isinstance(entries, dict):
        for k, v in entries.items():
            if isinstance(v, dict):
                v = v.get("library_path") or v.get("lib") or ""
            if isinstance(v, str) and v.strip():
                out[str(k).strip().strip("{}").lower()] = v.strip()
    return out


def save_replacements(pkg: Path, mapping: dict):
    """Merge `mapping` (key -> 'lib/...' | 'SKIP' | '') into the existing
    object_replacements.json; an empty value removes that key."""
    existing = load_existing(pkg)
    for k, v in mapping.items():
        k = str(k).strip().strip("{}").lower()
        v = (v or "").strip()
        if not v:
            existing.pop(k, None)
        else:
            existing[k] = v
    payload = {
        "_comment": "key -> X-Plane 'lib/...' path (or 'SKIP' to place nothing). "
                    "key is a placement GUID (no braces, lower-case) or its "
                    "normalized title. Written by pick_replacements.py, read by "
                    "the converter's resolve_library_substitution().",
        "replacements": dict(sorted(existing.items())),
    }
    path = pkg / "object_replacements.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path, len(existing)


# --------------------------------------------------------------------------
# X-Plane library scan
# --------------------------------------------------------------------------
_EXPORT_RE = re.compile(r"^\s*EXPORT(?:_EXCLUDE|_RATIO|_BACKUP|_EXTEND)?\s+(?:[\d.]+\s+)?(lib/\S+)", re.IGNORECASE)


def _resolve_xplane_root(cli_value):
    for cand in (cli_value, os.environ.get("MSFS2XP_XPLANE_ROOT"), "/xp11"):
        if cand and Path(cand).is_dir():
            return Path(cand)
    return None


def scan_library_paths(xp_root: Path):
    """Sorted unique list of every 'lib/...' virtual path EXPORTed by any
    library.txt under the install's default scenery + custom scenery."""
    if xp_root is None:
        return [], 0
    seen = set()
    files = 0
    for base in (xp_root / "Resources" / "default scenery", xp_root / "Custom Scenery"):
        if not base.is_dir():
            continue
        for lib in base.rglob("library.txt"):
            files += 1
            try:
                text = lib.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line in text.splitlines():
                m = _EXPORT_RE.match(line)
                if m:
                    seen.add(m.group(1))
    return sorted(seen), files


# --------------------------------------------------------------------------
# map preview
# --------------------------------------------------------------------------
_BASE_W = 1600  # px width the base airport image is rendered at


class _MapView:
    """A Tk canvas showing a top-down map of the converted airport (grey
    pavement outlines + faint dots for objects that DID place, both from
    the converter's `unresolved_map.json`) with the currently-selected
    unresolved object's placements drawn on top in red. The base image is
    rasterised once on a worker thread (pure PIL, no Tk); per-selection
    redraws are cheap (blit + a few dots + a crop)."""

    def __init__(self, parent, tk, map_base, all_positions):
        self._map_base = map_base or {}
        self._all_pos = [tuple(p) for p in all_positions if p]
        self._base = None          # PIL image, full airport extent (set by worker)
        self._bounds = None        # (la0, la1, lo0, lo1)  (set by worker)
        self._err = None           # str, if the worker failed
        self._sel = ([], "")       # (positions, label)
        self._photo = None         # keep a ref or Tk drops it
        self.frame = tk.Frame(parent)
        self.canvas = tk.Canvas(self.frame, bg="#0c0d10", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self._redraw())
        if not _HAVE_PIL:
            self._msg("Map preview needs Pillow (pip install Pillow). The picker still works.")
            return
        self._msg("Drawing airport map…")
        threading.Thread(target=self._build_base, daemon=True).start()
        self.frame.after(120, self._poll_ready)     # main-thread poll; worker never touches Tk

    def _msg(self, text):
        self.canvas.delete("all")
        self.canvas.create_text(12, 14, anchor="nw", fill="#999", text=text,
                                font=("TkDefaultFont", 10))

    def _poll_ready(self):
        if self._base is not None:
            self._redraw()
        elif self._err is not None:
            self._msg(f"Map unavailable: {self._err}")
        else:
            self.frame.after(120, self._poll_ready)

    def _build_base(self):
        """Worker thread: pure PIL work only -- NO Tk calls. Results land
        in self._base / self._bounds / self._err; the main thread's
        _poll_ready picks them up."""
        try:
            import math
            pav = [tuple(p) for p in (self._map_base.get("pavement_pts") or []) if len(p) == 2]
            ctx = [tuple(p) for p in (self._map_base.get("context") or []) if len(p) == 2]
            bnd = self._map_base.get("bounds")
            pts = list(self._all_pos) + ctx + pav
            if bnd and len(bnd) == 4:
                la0, la1, lo0, lo1 = bnd
            elif len(pts) >= 2:
                la = [p[0] for p in pts]
                lo = [p[1] for p in pts]
                la0, la1, lo0, lo1 = min(la), max(la), min(lo), max(lo)
            else:
                self._err = "no map data yet -- re-run the conversion with this build"
                return
            span = max((la1 - la0), (lo1 - lo0))
            pad = span * 0.03 + 1e-4
            la0 -= pad
            la1 += pad
            lo0 -= pad
            lo1 += pad
            mlon = 111320 * math.cos(math.radians((la0 + la1) / 2))
            W = _BASE_W
            H = int(W * ((la1 - la0) * 110540) / max((lo1 - lo0) * mlon, 1e-9))
            H = max(500, min(H, 3200))
            img = Image.new("RGB", (W, H), "#0c0d10")
            px = img.load()

            def xy(lat, lon):
                return (int((lon - lo0) / (lo1 - lo0) * (W - 1)),
                        int((la1 - lat) / (la1 - la0) * (H - 1)))

            for lat, lon in pav:                       # pavement silhouette: 1px scatter
                x, y = xy(lat, lon)
                if 0 <= x < W and 0 <= y < H:
                    px[x, y] = (86, 90, 100)
            dr = ImageDraw.Draw(img)
            for lat, lon in ctx:                       # objects that DID place: faint dots
                x, y = xy(lat, lon)
                dr.ellipse([x - 1, y - 1, x + 1, y + 1], fill=(120, 122, 134))
            self._bounds = (la0, la1, lo0, lo1)
            self._base = img
        except Exception as e:  # never let the map crash the picker
            self._err = str(e) or e.__class__.__name__

    def show(self, positions, label):
        self._sel = ([tuple(p) for p in (positions or []) if p], label or "")
        self._redraw()

    def _redraw(self):
        if self._base is None or self._bounds is None:
            return
        cw = max(self.canvas.winfo_width(), 50)
        ch = max(self.canvas.winfo_height(), 50)
        la0, la1, lo0, lo1 = self._bounds
        W, H = self._base.size

        def xy(lat, lon):
            return ((lon - lo0) / (lo1 - lo0) * W, (la1 - lat) / (la1 - la0) * H)

        positions, label = self._sel
        im = self._base.copy()
        dr = ImageDraw.Draw(im, "RGBA")
        box = None
        if positions:
            xs = [xy(*p)[0] for p in positions]
            ys = [xy(*p)[1] for p in positions]
            for x, y in zip(xs, ys):
                dr.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(255, 45, 85, 255), outline=(0, 0, 0, 160))
            m = 90
            box = (max(0, min(xs) - m), max(0, min(ys) - m),
                   min(W, max(xs) + m), min(H, max(ys) + m))
            # keep a sane minimum crop so a single point isn't a blurry zoom
            if box[2] - box[0] < 360:
                cx = (box[0] + box[2]) / 2
                box = (max(0, cx - 180), box[1], min(W, cx + 180), box[3])
            if box[3] - box[1] < 260:
                cy = (box[1] + box[3]) / 2
                box = (box[0], max(0, cy - 130), box[2], min(H, cy + 130))
        crop = im.crop(tuple(int(v) for v in box)) if box else im
        scale = min(cw / crop.width, ch / crop.height)
        disp = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))))
        self._photo = ImageTk.PhotoImage(disp)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._photo)
        cap = label + (f"   ({len(positions)} placement(s))" if positions else "   -- no recorded positions")
        self.canvas.create_rectangle(0, 0, cw, 22, fill="#000000", outline="")
        self.canvas.create_text(8, 4, anchor="nw", fill="#eee", text=cap,
                                font=("TkDefaultFont", 9, "bold"))


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------
def _populate_picker(win, pkg: Path, xp_root, *, modal: bool):
    """Build the picker into `win` (a Tk root or a Toplevel).
    Left = the unresolved objects (select one) -- shows the friendly name
    where the converter resolved one. Centre = a top-down map with the
    selected object's placements. Right = a searchable list of every
    X-Plane 'lib/...' path; pick one (or Skip, or a hand-typed custom
    path) per object; Save writes object_replacements.json.
    modal=True -> footer "Save & continue" / "Skip all & continue"; both
    close the window (used mid-conversion). modal=False -> "Save"/"Close"."""
    import tkinter as tk
    from tkinter import ttk, messagebox

    unresolved = load_unresolved(pkg)
    existing = load_existing(pkg)
    candidates, n_libfiles = scan_library_paths(xp_root)

    head = ttk.Frame(win, padding=8)
    head.pack(fill="x")
    ttk.Label(head, text=f"Package: {pkg}", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
    if not unresolved:
        ttk.Label(head, text="No unresolved_objects.json (or it's empty) -- nothing to map.").pack(anchor="w", pady=4)
        ttk.Button(head, text="Close", command=win.destroy).pack(anchor="w")
        return
    if modal:
        ttk.Label(head, text="These placements matched no converted geometry and no substitution. "
                             "Assign an X-Plane library object or Skip -- your picks are applied to "
                             "THIS conversion before the DSF is compiled, and saved for next time.",
                  wraplength=1400, foreground="#555").pack(anchor="w", pady=(2, 0))
    ttk.Label(head, text=f"{len(unresolved)} unresolved object(s)   |   "
                         f"{len(candidates)} X-Plane library path(s) from {n_libfiles} library.txt file(s)"
                         f"{'   |   X-PLANE ROOT NOT FOUND -- type paths by hand' if not candidates else ''}"
             ).pack(anchor="w")

    assign = {o["key"]: existing.get(o["key"], "") for o in unresolved}
    pos_by_key = {o["key"]: o["positions"] for o in unresolved}
    name_by_key = {o["key"]: (o["title"] or o["guid"] or o["key"]) for o in unresolved}

    panes = ttk.PanedWindow(win, orient="horizontal")
    panes.pack(fill="both", expand=True, padx=8, pady=4)

    # ---- left: unresolved objects -----------------------------------
    left = ttk.Frame(panes)
    panes.add(left, weight=3)
    ttk.Label(left, text="Unresolved objects", font=("TkDefaultFont", 9, "bold")).pack(anchor="w")
    tv = ttk.Treeview(left, columns=("count", "assign"), show="tree headings", height=26)
    tv.heading("#0", text="object")
    tv.heading("count", text="uses")
    tv.heading("assign", text="replacement")
    tv.column("#0", width=300, stretch=False)
    tv.column("count", width=50, anchor="e", stretch=False)
    tv.column("assign", width=300)
    tvsb = ttk.Scrollbar(left, orient="vertical", command=tv.yview)
    tv.configure(yscrollcommand=tvsb.set)
    tv.pack(side="left", fill="both", expand=True)
    tvsb.pack(side="right", fill="y")

    def _assign_text(v):
        return "— skip (place nothing) —" if v.upper() == SKIP else (v or "")

    for o in unresolved:
        tv.insert("", "end", iid=o["key"], text=name_by_key[o["key"]],
                  values=(o["count"], _assign_text(assign[o["key"]])))

    # ---- centre: map ----------------------------------------------
    centre = ttk.Frame(panes)
    panes.add(centre, weight=4)
    ttk.Label(centre, text="Where it is  (red = this object's placements)",
              font=("TkDefaultFont", 9, "bold")).pack(anchor="w")
    all_pos = [p for o in unresolved for p in o["positions"]]
    mapv = _MapView(centre, tk, load_map_base(pkg), all_pos)
    mapv.frame.pack(fill="both", expand=True)

    # ---- right: searchable X-Plane library list ------------------
    right = ttk.Frame(panes)
    panes.add(right, weight=3)
    ttk.Label(right, text="X-Plane library replacement", font=("TkDefaultFont", 9, "bold")).pack(anchor="w")
    sb = ttk.Frame(right)
    sb.pack(fill="x", pady=(2, 4))
    ttk.Label(sb, text="Search:").pack(side="left")
    search_var = tk.StringVar()
    search_ent = ttk.Entry(sb, textvariable=search_var)
    search_ent.pack(side="left", fill="x", expand=True, padx=4)

    lb_frame = ttk.Frame(right)
    lb_frame.pack(fill="both", expand=True)
    lb = tk.Listbox(lb_frame, activestyle="dotbox")
    lbsb = ttk.Scrollbar(lb_frame, orient="vertical", command=lb.yview)
    lb.configure(yscrollcommand=lbsb.set)
    lb.pack(side="left", fill="both", expand=True)
    lbsb.pack(side="right", fill="y")

    def _refilter(*_):
        q = search_var.get().strip().lower()
        terms = q.split()
        shown = [c for c in candidates if all(t in c.lower() for t in terms)] if terms else list(candidates)
        lb.delete(0, "end")
        for c in shown[:2000]:
            lb.insert("end", c)
        cnt.config(text=f"{len(shown)} match(es)" + ("  (first 2000 shown)" if len(shown) > 2000 else ""))
    search_var.trace_add("write", _refilter)

    cnt = ttk.Label(right, text="")
    cnt.pack(anchor="w")

    cf = ttk.Frame(right)
    cf.pack(fill="x", pady=(4, 0))
    ttk.Label(cf, text="or custom:").pack(side="left")
    custom_var = tk.StringVar()
    ttk.Entry(cf, textvariable=custom_var).pack(side="left", fill="x", expand=True, padx=4)

    def _sel_key():
        s = tv.selection()
        return s[0] if s else None

    def _apply(value):
        k = _sel_key()
        if not k:
            status.config(text="Pick an object row on the left first.")
            return
        assign[k] = value
        tv.set(k, "assign", _assign_text(value))
        rows_order = [o["key"] for o in unresolved]
        i = rows_order.index(k)
        for nk in rows_order[i + 1:]:
            if not assign[nk]:
                tv.selection_set(nk)
                tv.see(nk)
                break
        _status_counts()

    def _assign_selected(_evt=None):
        if custom_var.get().strip():
            _apply(custom_var.get().strip())
            return
        sel = lb.curselection()
        if not sel:
            status.config(text="Select a library path (or type a custom one).")
            return
        _apply(lb.get(sel[0]))

    lb.bind("<Double-Button-1>", _assign_selected)
    lb.bind("<Return>", _assign_selected)

    btns = ttk.Frame(right)
    btns.pack(fill="x", pady=6)
    ttk.Button(btns, text="Assign to selected object  ->", command=_assign_selected).pack(side="left")
    ttk.Button(btns, text="Skip", command=lambda: _apply(SKIP)).pack(side="left", padx=4)
    ttk.Button(btns, text="Clear", command=lambda: _apply("")).pack(side="left")

    def _on_select(_evt=None):
        k = _sel_key()
        if k is not None:
            mapv.show(pos_by_key.get(k, []), name_by_key.get(k, k))
    tv.bind("<<TreeviewSelect>>", _on_select)

    # ---- footer ---------------------------------------------------
    foot = ttk.Frame(win, padding=8)
    foot.pack(fill="x")
    status = ttk.Label(foot, text="")
    status.pack(side="left")

    def _status_counts():
        a = sum(1 for v in assign.values() if v and v.upper() != SKIP)
        s = sum(1 for v in assign.values() if v.upper() == SKIP)
        u = len(assign) - a - s
        status.config(text=f"{a} assigned   {s} skipped   {u} unset")

    def _save(close_after):
        path, n = save_replacements(pkg, assign)
        _status_counts()
        if close_after:
            win.destroy()
        else:
            messagebox.showinfo("Saved", f"Wrote {path}\n{n} replacement(s) stored.\n\n"
                                         f"They apply on the next conversion of this package.")

    if modal:
        ttk.Button(foot, text="Save & continue conversion", command=lambda: _save(True)).pack(side="right", padx=4)
        ttk.Button(foot, text="Skip all & continue", command=win.destroy).pack(side="right", padx=4)
    else:
        ttk.Button(foot, text="Save", command=lambda: _save(False)).pack(side="right", padx=4)
        ttk.Button(foot, text="Close", command=win.destroy).pack(side="right", padx=4)

    tv.selection_set(unresolved[0]["key"])
    _on_select()
    _refilter()
    _status_counts()
    search_ent.focus_set()


def prompt_modal(parent, pkg: Path, xp_root):
    """Modal picker on top of an already-running Tk app (the converter's
    own window), shown mid-conversion just before DSF compile. Blocks the
    caller until the user closes it. Returns True if there was anything to
    pick (window was shown), False if nothing was unresolved."""
    import tkinter as tk

    if not load_unresolved(pkg):
        return False
    top = tk.Toplevel(parent)
    top.title(f"Unresolved objects -- {pkg.name}")
    top.geometry("1580x880")
    top.transient(parent)
    _populate_picker(top, pkg, xp_root, modal=True)
    try:
        top.grab_set()
    except tk.TclError:
        pass
    top.protocol("WM_DELETE_WINDOW", top.destroy)
    parent.wait_window(top)
    return True


def run_gui(pkg: Path, xp_root):
    """Standalone tool entry point."""
    import tkinter as tk

    root = tk.Tk()
    root.title(f"Replacement picker -- {pkg.name}")
    root.geometry("1580x880")
    _populate_picker(root, pkg, xp_root, modal=False)
    root.mainloop()


def run_list(pkg: Path, xp_root: Path):
    unresolved = load_unresolved(pkg)
    existing = load_existing(pkg)
    candidates, n_libfiles = scan_library_paths(xp_root)
    print(f"Package: {pkg}")
    print(f"X-Plane root: {xp_root if xp_root else '(not found)'}")
    _mb = load_map_base(pkg)
    print(f"Map base: {len(_mb['pavement'])} pavement outline(s)" if _mb else "Map base: (not recorded)")
    print(f"Unresolved objects: {len(unresolved)}")
    for o in unresolved:
        cur = existing.get(o["key"], "")
        tag = f"  -> {cur}" if cur else ""
        gid = f"  [{o['guid']}]" if o["guid"] else ""
        pos = f"  {len(o['positions'])} pos" if o["positions"] else ""
        print(f"  - {o['title'] or o['key']}{gid}  x{o['count']}{pos}{tag}")
    print(f"X-Plane library candidates: {len(candidates)} (from {n_libfiles} library.txt file(s))")
    for c in candidates[:8]:
        print(f"    e.g. {c}")
    if existing:
        print(f"Existing object_replacements.json: {len(existing)} pick(s)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pick X-Plane library replacements for unresolved MSFS objects.")
    ap.add_argument("package", nargs="?", default=os.environ.get("MSFS2XP_PKG", _DEFAULT_PKG),
                    help="MSFS package dir (contains unresolved_objects.json)")
    ap.add_argument("--xplane", default=None, help="X-Plane install root (else $MSFS2XP_XPLANE_ROOT)")
    ap.add_argument("--list", action="store_true", help="print and exit, no GUI")
    args = ap.parse_args(argv)

    pkg = Path(args.package).expanduser()
    xp_root = _resolve_xplane_root(args.xplane)
    if not pkg.is_dir():
        print(f"error: package dir not found: {pkg}", file=sys.stderr)
        return 2
    if args.list:
        run_list(pkg, xp_root)
    else:
        run_gui(pkg, xp_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
