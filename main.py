import os
import sys
import json
import threading
import multiprocessing
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

from PIL import Image, ImageTk

import app_paths
import cache_utils
import pick_replacements
from pipeline import LOG_FILE_NAME, PipelineOptions, PipelineHooks, app_version, run_pipeline, wipe_cache_and_temp

# Settings live with the cache (next to the program, or the per-user
# folder when that's read-only -- see app_paths). An older build kept them
# in the working directory; that copy is still read if there's no new one.
CONFIG_FILE = app_paths.config_file()
_LEGACY_CONFIG_FILE = Path("msfs2xp_config.json")
TRANSPARENT_KEY = "#000001"  # Color key used for window corner rounding transparency



def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller bundle."""
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


def force_taskbar_icon(root):
    """Forces Windows OS DWM to render the frameless window on the Taskbar immediately."""
    if sys.platform == "win32":
        try:
            import ctypes
            root.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
            if hwnd == 0:
                hwnd = root.winfo_id()

            GWL_EXSTYLE = -20
            WS_EX_APPWINDOW = 0x00040000
            WS_EX_TOOLWINDOW = 0x00000080

            # Strip TOOLWINDOW flag and assign APPWINDOW flag
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            style = (style & ~WS_EX_TOOLWINDOW) | WS_EX_APPWINDOW
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)

            # Force Windows Shell to refresh frame state immediately
            SWP_NOSIZE = 0x0001
            SWP_NOMOVE = 0x0002
            SWP_NOZORDER = 0x0004
            SWP_FRAMECHANGED = 0x0020
            ctypes.windll.user32.SetWindowPos(
                hwnd, 0, 0, 0, 0, 0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED
            )
        except Exception:
            pass


def set_app_icon_and_taskbar(root, icon_filename="iconfin.ico"):
    """Applies window icon and sets Windows Process AppUserModelID."""
    icon_path = get_resource_path(icon_filename)

    if os.path.exists(icon_path):
        # wm iconbitmap only understands the native .ico format on Windows.
        # On X11 it can misparse the file's raw bytes and hand Xlib an
        # oversized icon-pixmap request directly -- a BadLength protocol
        # error that bypasses Tcl/Python exception handling entirely and
        # kills the process, so only try it on win32; everywhere else go
        # straight to the PIL-decoded iconphoto path, which uses real pixel
        # data and works cross-platform.
        icon_set = False
        if sys.platform == "win32":
            try:
                root.iconbitmap(icon_path)
                icon_set = True
            except Exception:
                icon_set = False
        if not icon_set:
            try:
                img = Image.open(icon_path)
                # iconfin.ico only embeds a single 2048x2048 master frame.
                # Handing that straight to iconphoto sends an X11
                # _NET_WM_ICON property update that big-requests-unaware
                # servers (e.g. Xvfb) reject as BadLength -- a protocol
                # error that bypasses Tcl/Python's own exception handling,
                # so downscale to a normal icon size first.
                if max(img.size) > 256:
                    img = img.resize((256, 256), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                root.iconphoto(True, photo)
            except Exception:
                pass

    if sys.platform == "win32":
        try:
            import ctypes
            my_app_id = "msfs2xp.converter.1.0"
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(my_app_id)
        except Exception:
            pass

    force_taskbar_icon(root)


# --- CANVAS ROUNDED GRAPHICS HELPERS ---
def draw_rounded_polygon(canvas, x1, y1, x2, y2, radius=10, **kwargs):
    """Draws a smooth rounded rectangle on a Tkinter Canvas."""
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    radius = max(2.0, min(radius, (x2 - x1) / 2.0, (y2 - y1) / 2.0))
    points = [
        x1 + radius, y1,
        x2 - radius, y1,
        x2, y1,
        x2, y1 + radius,
        x2, y2 - radius,
        x2, y2,
        x2 - radius, y2,
        x1 + radius, y2,
        x1, y2,
        x1, y2 - radius,
        x1, y1 + radius,
        x1, y1
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)


# --- CUSTOM ROUNDED WIDGETS ---
class RoundedCard(tk.Canvas):
    """A dark card container with smooth rounded corners."""
    def __init__(self, parent, bg_color="#141414", card_bg="#1c1c1e", border_color="#2a2a2e", radius=12, **kwargs):
        super().__init__(parent, bg=bg_color, highlightthickness=0, bd=0, **kwargs)
        self.bg_color = bg_color
        self.card_bg = card_bg
        self.border_color = border_color
        self.radius = radius
        
        self.inner = tk.Frame(self, bg=card_bg)
        self.bind("<Configure>", self._on_resize)

    def _on_resize(self, event):
        self.delete("card_bg")
        w, h = event.width, event.height
        if w > 2 and h > 2:
            draw_rounded_polygon(self, 1, 1, w - 1, h - 1, self.radius,
                                 fill=self.card_bg, outline=self.border_color, width=1, tags="card_bg")
            self.create_window(w / 2, h / 2, window=self.inner, width=w - 24, height=h - 24)


class RoundedButton(tk.Canvas):
    """A custom interactive button with smooth rounded borders."""
    def __init__(self, parent, text="", command=None, radius=8, bg_color="#1c1c1e",
                 btn_bg="#28282e", hover_bg="#38383e", active_bg="#1f1f24", disabled_bg="#18181a",
                 fg="#ffffff", disabled_fg="#55555e", font=("Segoe UI", 9, "bold"), width=85, height=32, **kwargs):
        super().__init__(parent, width=width, height=height, bg=bg_color, highlightthickness=0, bd=0, cursor="hand2", **kwargs)
        self.command = command
        self.radius = radius
        self.btn_bg = btn_bg
        self.hover_bg = hover_bg
        self.active_bg = active_bg
        self.disabled_bg = disabled_bg
        self.curr_bg = btn_bg
        self.fg = fg
        self.disabled_fg = disabled_fg
        self.text_str = text
        self.font = font
        self.state = "normal"
        
        self.bind("<Configure>", self.redraw)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)

    def redraw(self, event=None):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w > 2 and h > 2:
            bg = self.curr_bg if self.state == "normal" else self.disabled_bg
            fg = self.fg if self.state == "normal" else self.disabled_fg
            draw_rounded_polygon(self, 1, 1, w - 1, h - 1, self.radius, fill=bg, outline=bg)
            self.create_text(w / 2, h / 2, text=self.text_str, fill=fg, font=self.font)

    def _on_enter(self, e):
        if self.state == "normal":
            self.curr_bg = self.hover_bg
            self.redraw()

    def _on_leave(self, e):
        if self.state == "normal":
            self.curr_bg = self.btn_bg
            self.redraw()

    def _on_press(self, e):
        if self.state == "normal":
            self.curr_bg = self.active_bg
            self.redraw()

    def _on_release(self, e):
        if self.state == "normal":
            self.curr_bg = self.hover_bg
            self.redraw()
            if self.command:
                self.command()

    def set_state(self, state):
        self.state = state
        self.config(cursor="hand2" if state == "normal" else "arrow")
        self.redraw()


class RoundedEntry(tk.Canvas):
    """An input field enclosed within a rounded border box."""
    def __init__(self, parent, textvariable=None, radius=8, bg_color="#1c1c1e",
                 entry_bg="#101012", border_color="#2a2a2e", fg="#ffffff", font=("Segoe UI", 9), **kwargs):
        super().__init__(parent, height=32, bg=bg_color, highlightthickness=0, bd=0, **kwargs)
        self.radius = radius
        self.entry_bg = entry_bg
        self.border_color = border_color
        
        self.entry = tk.Entry(self, textvariable=textvariable, bg=entry_bg, fg=fg,
                              insertbackground=fg, bd=0, font=font, highlightthickness=0)
        self.bind("<Configure>", self._on_resize)

    def _on_resize(self, event):
        self.delete("all")
        w, h = event.width, event.height
        if w > 2 and h > 2:
            draw_rounded_polygon(self, 1, 1, w - 1, h - 1, self.radius, fill=self.entry_bg, outline=self.border_color, width=1)
            self.create_window(12, h / 2, window=self.entry, anchor="w", width=w - 24)


class RoundedProgressBar(tk.Canvas):
    """A pill-shaped smooth custom progress bar."""
    def __init__(self, parent, height=14, radius=7, bg_color="#1c1c1e",
                 trough_color="#101012", bar_color="#2563eb", **kwargs):
        super().__init__(parent, height=height, bg=bg_color, highlightthickness=0, bd=0, **kwargs)
        self.radius = radius
        self.trough_color = trough_color
        self.bar_color = bar_color
        self.value = 0
        self.maximum = 100
        self.mode = "determinate"
        self._anim_pos = 0
        self._anim_job = None
        self.bind("<Configure>", self.redraw)

    def set_progress(self, val, max_val=100):
        self.stop_animation()
        self.mode = "determinate"
        self.value = val
        self.maximum = max_val
        self.redraw()

    def set_color(self, color):
        self.bar_color = color
        self.redraw()

    def start_indeterminate(self):
        self.mode = "indeterminate"
        if not self._anim_job:
            self._animate()

    def stop_animation(self):
        if self._anim_job:
            self.after_cancel(self._anim_job)
            self._anim_job = None

    def _animate(self):
        if self.mode == "indeterminate":
            self._anim_pos = (self._anim_pos + 4) % 100
            self.redraw()
            self._anim_job = self.after(30, self._animate)

    def redraw(self, event=None):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w <= 2 or h <= 2:
            return
            
        r = min(self.radius, h / 2)
        draw_rounded_polygon(self, 0, 0, w, h, r, fill=self.trough_color, outline="")
        
        if self.mode == "determinate":
            if self.maximum > 0 and self.value > 0:
                pct = min(1.0, max(0.0, self.value / float(self.maximum)))
                pw = max(r * 2, int(w * pct))
                draw_rounded_polygon(self, 0, 0, min(w, pw), h, r, fill=self.bar_color, outline="")
        else:
            bw = int(w * 0.3)
            start_x = int((w + bw) * (self._anim_pos / 100.0)) - bw
            end_x = start_x + bw
            
            x1 = max(0, start_x)
            x2 = min(w, end_x)
            if x2 - x1 > r:
                draw_rounded_polygon(self, x1, 0, x2, h, r, fill=self.bar_color, outline="")


# --- MAIN APP CLASS ---
class ModularPythonConverterApp:
    def __init__(self, root):
        self.root = root
        
        # Hide window initially to prevent taskbar glitches
        self.root.withdraw()
        
        self.root.overrideredirect(True)
        self.root.geometry("1240x758")
        self.root.minsize(1100, 638)
        
        self.pkg_var = tk.StringVar()
        self.out_var = tk.StringVar()
        self.msfs_var = tk.StringVar()  # optional: MSFS install root, for stock/ASOBO object resolution
        # Optional: your own copy of the MSFS SDK's "Propdefs" folder,
        # needed to decode SimPropContainer-based placements (apron
        # lights/lamps, jetways, animated doors, building interiors). Not
        # bundled with this app -- those are Microsoft/Asobo's own SDK
        # data files, not this project's to redistribute.
        self.propdefs_var = tk.StringVar()
        # xp12: modern X-Plane 11.50+/XP12 apt.dat spec (1200, the default
        # apt_dat.write_apt_dat already writes as the ACTIVE "apt.dat").
        # xp11: legacy pre-11.50 X-Plane 11 -- write_apt_dat already
        # produces this exact file too (spec 1100, jetway rows stripped),
        # today only as a ".xp11" sidecar the user has to manually rename
        # over apt.dat; this toggle makes it the active file instead when
        # the user's actual target is an older XP11 install.
        self.xp_version_var = tk.StringVar(value="xp12")
        self.clean_run_var = tk.BooleanVar(value=False)
        self.disable_cache_var = tk.BooleanVar(value=False)
        # Proximity-triggered animations (doors that open as the aircraft
        # approaches) need the companion msfs2xp_proximity_animator
        # FlyWithLua script to actually move -- without it, or while the
        # animated geometry is still misplaced, render them as static
        # rigid geometry at rest pose instead (see mesh_convert.convert's
        # disable_proximity_animation parameter). Default ON.
        self.static_doors_var = tk.BooleanVar(value=True)
        # See resolve_library_substitution's own docstring for what this
        # does (a local string-similarity heuristic, not a hosted service).
        self.approximate_substitution_var = tk.BooleanVar(value=False)
        # Scans BGL TerrainVectorDb sections (MSFS World Editor's vector-
        # polygon/vegetation database -- some packages, e.g. iniBuilds'
        # EGLC, store their whole taxiway/runway/apron pavement here
        # instead of as placed glTF models) and logs which ground
        # materials each polygon references. Diagnostic-only (material
        # references, not geometry) -- see bgl_extractor.scan_terrain_
        # vector_db's own docstring for what was and wasn't decoded.
        self.scan_terrain_vectors_var = tk.BooleanVar(value=True)
        # Opt-in .pol/DSF-polygon rollout: draped ground content (pavement
        # fills, painted markings/signs) as real X-Plane .pol draped
        # polygons via native DSF primitives instead of OBJ8 .obj files
        # with ATTR_draped -- see draped_merge.merge_draped_layers_in_
        # tile's own docstring. Off keeps the existing OBJ8 path, which
        # stays a permanent fallback, not a temporary one.
        self.pol_polygons_var = tk.BooleanVar(value=False)
        # When on (default), a conversion that hits objects it can't match
        # to any converted geometry or substitution pauses just before DSF
        # compile and pops up the replacement picker (pick_replacements) so
        # the user can map them to X-Plane library objects / skip them, and
        # those picks are applied to THIS run. Headless runners set this
        # False (there's no display / event loop to host the dialog).
        self.prompt_replacements_var = tk.BooleanVar(value=True)
        # apt.dat from the package's own airport record: draw real X-Plane
        # runways (instead of a transparent hard surface under the converted
        # pavement), and paint the MSFS painted-line records.
        self.native_runways_var = tk.BooleanVar(value=False)
        self.native_painted_lines_var = tk.BooleanVar(value=False)
        self.flatten_airport_var = tk.BooleanVar(value=False)
        self.remove_runway_clutter_var = tk.BooleanVar(value=True)
        self.exclusions_var = tk.BooleanVar(value=True)
        self.write_log_file_var = tk.BooleanVar(value=False)
        self.no_autodgs_var = tk.BooleanVar(value=True)
        # How opaque blended MSFS glass is drawn (MSFS glass relies on
        # reflections X-Plane doesn't draw, so its own alpha is near zero).
        self.glass_opacity_var = tk.IntVar(value=50)
        # Largest texture side for the biggest buildings (smaller objects get
        # proportionally less); 0 keeps source sizes.
        self.max_texture_var = tk.IntVar(value=2048)
        self.is_maximized = False
        self._is_minimizing = False
        
        if sys.platform == "win32":
            try:
                self.root.wm_attributes("-transparentcolor", TRANSPARENT_KEY)
            except Exception:
                pass
        self.root.configure(bg=TRANSPARENT_KEY)

        set_app_icon_and_taskbar(self.root, "iconfin.ico")

        self.build_custom_window_shell()
        self.build_ui()
        self.load_config()

        # Reveal window and force taskbar refresh after window map initialization
        self.root.deiconify()
        self.root.after(50, lambda: force_taskbar_icon(self.root))

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<Map>", self._on_window_map)

    def build_custom_window_shell(self):
        self.window_canvas = tk.Canvas(self.root, bg=TRANSPARENT_KEY, highlightthickness=0, bd=0)
        self.window_canvas.pack(fill=tk.BOTH, expand=True)

        self.bg_dark = "#141414"
        self.card_bg = "#1c1c1e"
        self.title_bg = "#181818"
        self.accent_blue = "#2563eb"
        self.border_color = "#2a2a2e"

        self.main_container = tk.Frame(self.window_canvas, bg=self.bg_dark)
        
        self.window_canvas.bind("<Configure>", self._draw_window_shape)

        # --- CUSTOM TITLE BAR ---
        self.title_bar = tk.Frame(self.main_container, bg=self.title_bg, height=38)
        self.title_bar.pack(fill=tk.X, side=tk.TOP)
        self.title_bar.pack_propagate(False)

        dots_frame = tk.Frame(self.title_bar, bg=self.title_bg)
        dots_frame.pack(side=tk.LEFT, padx=14)

        def create_dot(parent, color, hover_color, command):
            canvas = tk.Canvas(parent, width=12, height=12, bg=self.title_bg, highlightthickness=0, cursor="hand2")
            dot = canvas.create_oval(1, 1, 11, 11, fill=color, outline=color)
            canvas.bind("<Enter>", lambda e: canvas.itemconfig(dot, fill=hover_color, outline=hover_color))
            canvas.bind("<Leave>", lambda e: canvas.itemconfig(dot, fill=color, outline=color))
            canvas.bind("<Button-1>", lambda e: command())
            return canvas

        self.close_dot = create_dot(dots_frame, "#ff5f56", "#e0443e", self.on_close)
        self.close_dot.pack(side=tk.LEFT, padx=3)

        self.min_dot = create_dot(dots_frame, "#ffbd2e", "#dea123", self.minimize_window)
        self.min_dot.pack(side=tk.LEFT, padx=3)

        self.max_dot = create_dot(dots_frame, "#27c93f", "#1aab29", self.toggle_maximize)
        self.max_dot.pack(side=tk.LEFT, padx=3)

        self.title_label = tk.Label(self.title_bar, text="MSFS2XP", bg=self.title_bg,
                                    fg="#a0a0a0", font=("Segoe UI", 9, "bold"))
        self.title_label.pack(side=tk.LEFT, padx=12)

        self.version_label = tk.Label(self.title_bar, text=_version_text(), bg=self.title_bg,
                                      fg="#55555e", font=("Segoe UI", 8))
        self.version_label.pack(side=tk.RIGHT, padx=14)

        for widget in (self.title_bar, self.title_label, self.version_label):
            widget.bind("<Button-1>", self._start_drag)
            widget.bind("<B1-Motion>", self._do_drag)
            widget.bind("<Double-Button-1>", lambda e: self.toggle_maximize())

        # "size_nw_se" is a Windows-only Tk cursor name; X11 (Linux) has no
        # such cursor and raises TclError on it, crashing the app at
        # startup. "bottom_right_corner" is the standard X cursor-font
        # glyph for the same NW-SE resize affordance.
        _grip_cursor = "size_nw_se" if sys.platform == "win32" else "bottom_right_corner"
        self.grip = tk.Label(self.main_container, text="◢", bg=self.bg_dark, fg="#33333e", cursor=_grip_cursor)
        self.grip.place(relx=1.0, rely=1.0, anchor="se")
        self.grip.bind("<Button-1>", self._start_resize)
        self.grip.bind("<B1-Motion>", self._do_resize)

    def _draw_window_shape(self, event):
        self.window_canvas.delete("win_bg")
        w, h = event.width, event.height
        if w > 2 and h > 2:
            r = 14 if not self.is_maximized else 0
            draw_rounded_polygon(self.window_canvas, 0, 0, w, h, radius=r,
                                 fill=self.bg_dark, outline=self.border_color, width=1, tags="win_bg")
            self.window_canvas.create_window(w / 2, h / 2, window=self.main_container, width=w - 2, height=h - 2)

    def _start_drag(self, event):
        self._drag_x = event.x
        self._drag_y = event.y

    def _do_drag(self, event):
        if self.is_maximized:
            return
        x = self.root.winfo_x() + (event.x - self._drag_x)
        y = self.root.winfo_y() + (event.y - self._drag_y)
        self.root.geometry(f"+{x}+{y}")

    def _start_resize(self, event):
        self._resize_x = event.x_root
        self._resize_y = event.y_root
        self._start_w = self.root.winfo_width()
        self._start_h = self.root.winfo_height()

    def _do_resize(self, event):
        delta_w = event.x_root - self._resize_x
        delta_h = event.y_root - self._resize_y
        new_w = max(800, self._start_w + delta_w)
        new_h = max(600, self._start_h + delta_h)
        self.root.geometry(f"{new_w}x{new_h}")

    def minimize_window(self):
        self._is_minimizing = True
        self.root.overrideredirect(False)
        self.root.iconify()

    def _on_window_map(self, event):
        if event.widget == self.root and self.root.state() == 'normal':
            if getattr(self, '_is_minimizing', False):
                self.root.overrideredirect(True)
                force_taskbar_icon(self.root)
                self._is_minimizing = False

    def toggle_maximize(self):
        if not self.is_maximized:
            self._prev_geom = self.root.geometry()
            sw = self.root.winfo_screenwidth()
            sh = self.root.winfo_screenheight()
            self.root.geometry(f"{sw}x{sh}+0+0")
            self.is_maximized = True
        else:
            if hasattr(self, '_prev_geom'):
                self.root.geometry(self._prev_geom)
            self.is_maximized = False
        self._draw_window_shape(type('Event', (), {'width': self.root.winfo_width(), 'height': self.root.winfo_height()})())

    def load_config(self):
        config_file = CONFIG_FILE if CONFIG_FILE.exists() else _LEGACY_CONFIG_FILE
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    data = json.load(f)
                    self.pkg_var.set(data.get("pkg_dir", ""))
                    self.out_var.set(data.get("out_dir", ""))
                    self.msfs_var.set(data.get("msfs_install_dir", ""))
                    self.propdefs_var.set(data.get("propdefs_dir", ""))
                    self.xp_version_var.set(data.get("xp_version", "xp12"))
                    self.clean_run_var.set(data.get("clean_run", False))
                    self.disable_cache_var.set(data.get("disable_cache", False))
                    # Default True here too, matching the BooleanVar's own
                    # declared default above -- an old saved config from
                    # before this key existed (or from a run where it was
                    # left unchecked) must not silently revert a returning
                    # user to animated, unreliable doors; "static except
                    # lights" is the intended default regardless of what an
                    # old config file does or doesn't have recorded.
                    self.static_doors_var.set(data.get("static_doors", True))
                    self.approximate_substitution_var.set(data.get("approximate_substitution", False))
                    self.scan_terrain_vectors_var.set(data.get("scan_terrain_vectors", True))
                    self.pol_polygons_var.set(data.get("pol_polygons", False))
                    self.native_runways_var.set(data.get("native_runways", False))
                    self.native_painted_lines_var.set(data.get("native_painted_lines", False))
                    self.flatten_airport_var.set(data.get("flatten_airport", False))
                    self.remove_runway_clutter_var.set(data.get("remove_runway_clutter", True))
                    self.exclusions_var.set(data.get("exclusions", True))
                    self.write_log_file_var.set(data.get("write_log_file", False))
                    self.no_autodgs_var.set(data.get("no_autodgs", True))
                    self.glass_opacity_var.set(int(data.get("glass_opacity", 50)))
                    self.max_texture_var.set(int(data.get("max_texture", 2048)))
                    geom = data.get("geometry")
                    if geom:
                        self.root.geometry(geom)
            except Exception:
                pass

    def save_config(self):
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump({
                    "pkg_dir": self.pkg_var.get(),
                    "out_dir": self.out_var.get(),
                    "msfs_install_dir": self.msfs_var.get(),
                    "propdefs_dir": self.propdefs_var.get(),
                    "xp_version": self.xp_version_var.get(),
                    "clean_run": self.clean_run_var.get(),
                    "disable_cache": self.disable_cache_var.get(),
                    "static_doors": self.static_doors_var.get(),
                    "approximate_substitution": self.approximate_substitution_var.get(),
                    "scan_terrain_vectors": self.scan_terrain_vectors_var.get(),
                    "pol_polygons": self.pol_polygons_var.get(),
                    "native_runways": self.native_runways_var.get(),
                    "native_painted_lines": self.native_painted_lines_var.get(),
                    "flatten_airport": self.flatten_airport_var.get(),
                    "remove_runway_clutter": self.remove_runway_clutter_var.get(),
                    "exclusions": self.exclusions_var.get(),
                    "write_log_file": self.write_log_file_var.get(),
                    "no_autodgs": self.no_autodgs_var.get(),
                    "glass_opacity": self._glass_opacity(),
                    "max_texture": self._max_texture(),
                    "geometry": self.root.geometry()
                }, f)
        except Exception:
            pass

    def on_close(self):
        self.save_config()
        self.root.destroy()

    def build_ui(self):
        main = tk.Frame(self.main_container, bg=self.bg_dark)
        main.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)

        # Two columns: the controls (setup / options / progress / execute)
        # stack down a fixed-width left column, and the run log gets its
        # own full-height panel on the right instead of being crammed
        # under everything. left_col has BOTH width and height pinned with
        # geometry propagation off, so its child cards can't drive it wider
        # and the split stays stable.
        body = tk.Frame(main, bg=self.bg_dark)
        body.pack(fill=tk.BOTH, expand=True)
        left_col = tk.Frame(body, bg=self.bg_dark, width=596, height=678)
        left_col.pack(side=tk.LEFT, fill=tk.Y)
        left_col.pack_propagate(False)
        right_col = tk.Frame(body, bg=self.bg_dark)
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(14, 0))

        # --- SETUP CARD ---
        setup_card = RoundedCard(left_col, bg_color=self.bg_dark, card_bg=self.card_bg, radius=12, height=201)
        setup_card.pack(fill=tk.X, pady=(0, 12))
        
        setup_inner = setup_card.inner
        setup_inner.columnconfigure(1, weight=1)

        tk.Label(setup_inner, text="SETUP", bg=self.card_bg, fg="#3b82f6", font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        tk.Label(setup_inner, text="Source:", bg=self.card_bg, fg="#a0a0a0", font=("Segoe UI", 9)).grid(
            row=1, column=0, sticky="e", padx=(0, 10), pady=4)
        
        self.src_entry = RoundedEntry(setup_inner, textvariable=self.pkg_var, bg_color=self.card_bg)
        self.src_entry.grid(row=1, column=1, sticky="we", padx=5, pady=4)
        
        RoundedButton(setup_inner, text="Browse...", command=lambda: self.pkg_var.set(filedialog.askdirectory()),
                      bg_color=self.card_bg, width=85, height=30).grid(row=1, column=2, padx=(5, 0), pady=4)

        tk.Label(setup_inner, text="Target:", bg=self.card_bg, fg="#a0a0a0", font=("Segoe UI", 9)).grid(
            row=2, column=0, sticky="e", padx=(0, 10), pady=4)
        
        self.tgt_entry = RoundedEntry(setup_inner, textvariable=self.out_var, bg_color=self.card_bg)
        self.tgt_entry.grid(row=2, column=1, sticky="we", padx=5, pady=4)
        
        RoundedButton(setup_inner, text="Browse...", command=lambda: self.out_var.set(filedialog.askdirectory()),
                      bg_color=self.card_bg, width=85, height=30).grid(row=2, column=2, padx=(5, 0), pady=4)

        tk.Label(setup_inner, text="MSFS 2020 (optional):", bg=self.card_bg, fg="#a0a0a0", font=("Segoe UI", 9)).grid(
            row=3, column=0, sticky="e", padx=(0, 10), pady=4)

        self.msfs_entry = RoundedEntry(setup_inner, textvariable=self.msfs_var, bg_color=self.card_bg)
        self.msfs_entry.grid(row=3, column=1, sticky="we", padx=5, pady=4)

        RoundedButton(setup_inner, text="Browse...", command=lambda: self.msfs_var.set(filedialog.askdirectory()),
                      bg_color=self.card_bg, width=85, height=30).grid(row=3, column=2, padx=(5, 0), pady=4)

        tk.Label(setup_inner, text="Propdefs folder (optional):", bg=self.card_bg, fg="#a0a0a0", font=("Segoe UI", 9)).grid(
            row=4, column=0, sticky="e", padx=(0, 10), pady=4)

        # Your own copy of the MSFS SDK's "Propdefs" XML folder --
        # Microsoft/Asobo's own SDK data, not something this app can
        # legally bundle. Needed only to decode SimPropContainer-based
        # placements (many apron lights/lamps, jetways, animated doors,
        # building interiors); everything else converts fine without it.
        self.propdefs_entry = RoundedEntry(setup_inner, textvariable=self.propdefs_var, bg_color=self.card_bg)
        self.propdefs_entry.grid(row=4, column=1, sticky="we", padx=5, pady=4)

        RoundedButton(setup_inner, text="Browse...", command=lambda: self.propdefs_var.set(filedialog.askdirectory()),
                      bg_color=self.card_bg, width=85, height=30).grid(row=4, column=2, padx=(5, 0), pady=4)

        # --- OPTIONS CARD ---
        # Two pages in one fixed-height card: the options most runs touch,
        # and the rest under "Advanced".
        options_card = RoundedCard(left_col, bg_color=self.bg_dark, card_bg=self.card_bg, radius=12, height=222)
        options_card.pack(fill=tk.X, pady=(0, 12))

        options_inner = options_card.inner
        options_inner.columnconfigure(0, weight=1)

        def _styled_checkbutton(parent, text, variable):
            return tk.Checkbutton(
                parent, text=text, variable=variable, bg=self.card_bg, fg="#cccccc",
                activebackground=self.card_bg, activeforeground="#ffffff", selectcolor=self.bg_dark,
                highlightthickness=0, bd=0, font=("Segoe UI", 9), anchor="w")

        def _styled_radiobutton(parent, text, variable, value):
            return tk.Radiobutton(
                parent, text=text, variable=variable, value=value, bg=self.card_bg, fg="#cccccc",
                activebackground=self.card_bg, activeforeground="#ffffff", selectcolor=self.bg_dark,
                highlightthickness=0, bd=0, font=("Segoe UI", 9), anchor="w")

        header = tk.Frame(options_inner, bg=self.card_bg)
        header.grid(row=0, column=0, sticky="we", pady=(0, 6))
        tk.Label(header, text="OPTIONS", bg=self.card_bg, fg="#3b82f6", font=("Segoe UI", 10, "bold")).pack(
            side=tk.LEFT)

        main_page = tk.Frame(options_inner, bg=self.card_bg)
        advanced_page = tk.Frame(options_inner, bg=self.card_bg)
        for page in (main_page, advanced_page):
            page.grid(row=1, column=0, sticky="nwe")
            page.columnconfigure(3, weight=1)
        pages = {"Main": main_page, "Advanced": advanced_page}
        tabs = {}

        def show_page(name):
            for page_name, page in pages.items():
                if page_name == name:
                    page.grid()
                else:
                    page.grid_remove()
            for tab_name, tab in tabs.items():
                tab.config(fg="#ffffff" if tab_name == name else "#6b6b75")

        for name in ("Advanced", "Main"):  # packed right to left
            tab = tk.Label(header, text=name, bg=self.card_bg, fg="#6b6b75", cursor="hand2",
                           font=("Segoe UI", 9, "bold"))
            tab.pack(side=tk.RIGHT, padx=(10, 0))
            tab.bind("<Button-1>", lambda e, n=name: show_page(n))
            tabs[name] = tab

        # Main page
        row = 0
        tk.Label(main_page, text="Target X-Plane:", bg=self.card_bg, fg="#a0a0a0", font=("Segoe UI", 9)).grid(
            row=row, column=0, sticky="w", padx=(0, 10), pady=2)
        _styled_radiobutton(main_page, "XP11.50+ / XP12 (modern apt.dat)", self.xp_version_var, "xp12").grid(
            row=row, column=1, sticky="w", pady=2)
        _styled_radiobutton(main_page, "XP11 legacy (pre-11.50)", self.xp_version_var, "xp11").grid(
            row=row, column=2, columnspan=2, sticky="w", padx=(10, 0), pady=2)
        row += 1
        _styled_checkbutton(
            main_page, "Exclusion zones (hide X-Plane's own objects, forests and roads under the scenery)",
            self.exclusions_var
        ).grid(row=row, column=0, columnspan=4, sticky="w", pady=2)
        row += 1
        _styled_checkbutton(
            main_page, "Flat objects on the airport ground: remove small ones, drape large sheets",
            self.remove_runway_clutter_var
        ).grid(row=row, column=0, columnspan=4, sticky="w", pady=2)
        row += 1
        glass_row = tk.Frame(main_page, bg=self.card_bg)
        glass_row.grid(row=row, column=0, columnspan=4, sticky="w", pady=2)
        tk.Label(glass_row, text="Glass opacity % (100 = solid):", bg=self.card_bg, fg="#cccccc",
                 font=("Segoe UI", 9)).pack(side=tk.LEFT)
        tk.Spinbox(glass_row, from_=1, to=100, increment=5, width=5, textvariable=self.glass_opacity_var,
                   bg="#1c1c1e", fg="#ffffff", buttonbackground="#2a2a2e", relief=tk.FLAT,
                   insertbackground="#ffffff").pack(side=tk.LEFT, padx=(8, 0))
        tk.Label(glass_row, text="Max texture size px:", bg=self.card_bg, fg="#cccccc",
                 font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(16, 0))
        # A Spinbox with a list of values resets its variable to the first
        # one (0) when it is created: put the chosen size back afterwards.
        max_texture = self._max_texture()
        tk.Spinbox(glass_row, values=(0, 512, 1024, 2048, 4096, 8192), width=6, textvariable=self.max_texture_var,
                   bg="#1c1c1e", fg="#ffffff", buttonbackground="#2a2a2e", relief=tk.FLAT,
                   insertbackground="#ffffff").pack(side=tk.LEFT, padx=(8, 0))
        self.max_texture_var.set(max_texture)
        row += 1
        _styled_checkbutton(
            main_page, f"Write a .log file of the run into the output folder ({LOG_FILE_NAME})",
            self.write_log_file_var
        ).grid(row=row, column=0, columnspan=4, sticky="w", pady=2)
        row += 1
        cache_row = tk.Frame(main_page, bg=self.card_bg)
        cache_row.grid(row=row, column=0, columnspan=4, sticky="we", pady=2)
        _styled_checkbutton(cache_row, "Clean run (wipe cache + temp files first)", self.clean_run_var).pack(
            side=tk.LEFT)
        _styled_checkbutton(cache_row, "No cache this run", self.disable_cache_var).pack(side=tk.LEFT, padx=(14, 0))
        RoundedButton(cache_row, text="Clear Cache Now", command=self.clear_cache_now,
                      bg_color=self.card_bg, width=120, height=24).pack(side=tk.RIGHT)

        # Advanced page
        advanced = (
            ("Static doors (disable proximity- and business-hours-triggered animations)", self.static_doors_var),
            ("apt.dat: draw X-Plane runways with markings (for packages without runway models)",
             self.native_runways_var),
            ("apt.dat: paint the MSFS painted lines (if the draped models don't carry them)",
             self.native_painted_lines_var),
            ("apt.dat: flatten the terrain inside the airport, as MSFS does (if objects still float/sink)",
             self.flatten_airport_var),
            ("openSAM: no docking guidance of its own at this airport (writes no_autodgs.txt)",
             self.no_autodgs_var),
            ("Approximate name-matching for unresolved base-game objects (heuristic)",
             self.approximate_substitution_var),
            ("Convert draped pavement/markings to real .pol DSF polygons (experimental)", self.pol_polygons_var),
            ("Scan for TerrainVectorDb ground-polygon materials (diagnostic log only)",
             self.scan_terrain_vectors_var),
        )
        for i, (text, var) in enumerate(advanced):
            _styled_checkbutton(advanced_page, text, var).grid(row=i, column=0, columnspan=4, sticky="w", pady=2)

        show_page("Main")

        # --- PROGRESS CARD ---
        prog_card = RoundedCard(left_col, bg_color=self.bg_dark, card_bg=self.card_bg, radius=12, height=210)
        prog_card.pack(fill=tk.X, pady=(0, 12))
        
        prog_inner = prog_card.inner
        prog_inner.columnconfigure(1, weight=1)

        tk.Label(prog_inner, text="PROGRESS", bg=self.card_bg, fg="#3b82f6", font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))

        tk.Label(prog_inner, text="Overall pipeline:", bg=self.card_bg, fg="#cccccc", font=("Segoe UI", 9)).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(0, 4))
        
        self.overall_bar = RoundedProgressBar(prog_inner, height=16, radius=8, bg_color=self.card_bg, bar_color=self.accent_blue)
        self.overall_bar.grid(row=2, column=0, columnspan=2, sticky="we", pady=(0, 8))

        tk.Frame(prog_inner, bg="#2a2a2e", height=1).grid(row=3, column=0, columnspan=2, sticky="we", pady=(0, 8))

        step_frame = tk.Frame(prog_inner, bg=self.card_bg)
        step_frame.grid(row=4, column=0, columnspan=2, sticky="we")
        step_frame.columnconfigure(2, weight=1)

        def create_step_row(parent, number, label_text, row):
            tk.Label(parent, text=f"#{number}", bg=self.card_bg, fg="#3b82f6", font=("Segoe UI", 9, "bold"), width=3, anchor="w").grid(
                row=row, column=0, sticky="w", padx=(0, 4), pady=3)
            tk.Label(parent, text=label_text, bg=self.card_bg, fg="#888888", font=("Segoe UI", 9), width=16, anchor="w").grid(
                row=row, column=1, sticky="w", padx=(0, 10), pady=3)
            bar = RoundedProgressBar(parent, height=12, radius=6, bg_color=self.card_bg, bar_color=self.accent_blue)
            bar.grid(row=row, column=2, sticky="we", pady=3)
            return bar

        self.step1_bar = create_step_row(step_frame, 1, "BGL extraction", 0)
        self.step2_bar = create_step_row(step_frame, 2, "Texture extraction", 1)
        self.step3_bar = create_step_row(step_frame, 3, "Mesh conversion", 2)
        self.step4_bar = create_step_row(step_frame, 4, "DSF compilation", 3)

        # --- EXECUTE BUTTON ---
        btn_frame = tk.Frame(left_col, bg=self.bg_dark)
        btn_frame.pack(fill=tk.X, pady=(0, 12))
        
        self.start_btn = RoundedButton(btn_frame, text="EXECUTE", command=self.start, radius=10,
                                       bg_color=self.bg_dark, btn_bg=self.accent_blue, hover_bg="#1d4ed8",
                                       active_bg="#1e40af", fg="#ffffff", font=("Segoe UI", 10, "bold"),
                                       width=160, height=38)
        self.start_btn.pack()

        # --- LOGS CARD (own full-height panel, right column) ---
        log_card = RoundedCard(right_col, bg_color=self.bg_dark, card_bg=self.card_bg, radius=12)
        log_card.pack(fill=tk.BOTH, expand=True)
        
        log_inner = log_card.inner
        
        tk.Label(log_inner, text="Logs", bg=self.card_bg, fg="#3b82f6", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 6))

        self.log_area = ScrolledText(log_inner, bg="#0d0d0f", fg="#cccccc", font=("Consolas", 9),
                                     bd=0, highlightthickness=0, insertbackground="#ffffff")
        self.log_area.pack(fill=tk.BOTH, expand=True)

        self.log_area.tag_config("info", foreground="#cccccc")
        self.log_area.tag_config("success", foreground="#4ade80")
        self.log_area.tag_config("warning", foreground="#facc15")
        self.log_area.tag_config("error", foreground="#f87171")
        self.log_area.tag_config("header", foreground="#38bdf8", font=("Consolas", 10, "bold"))

    def _append_log(self, text, level):
        self.log_area.insert(tk.END, text + "\n", level)
        self.log_area.see(tk.END)

    def log(self, text, level="info"):
        self.root.after(0, self._append_log, text, level)
        
    def update_progress(self, bar, value, maximum=100):
        self.root.after(0, lambda: bar.set_progress(value, maximum))

    def stop_indeterminate(self, bar):
        self.root.after(0, lambda: bar.set_progress(0, 100))

    def mark_overall_complete(self):
        self.root.after(0, self._mark_overall_complete)

    def _mark_overall_complete(self):
        self.overall_bar.set_color("#22c55e")
        self.overall_bar.set_progress(100, 100)

    def reset_overall_bar(self):
        self.overall_bar.set_color(self.accent_blue)
        self.overall_bar.set_progress(0, 100)

    def set_indeterminate(self, bar):
        self.root.after(0, lambda: bar.start_indeterminate())

    def start(self):
        if not self.pkg_var.get() or not self.out_var.get():
            messagebox.showerror("Error", "Please provide both directories.")
            return
            
        self.save_config()
        self.start_btn.set_state("disabled")
        self.log_area.delete(1.0, tk.END)
        self.reset_overall_bar()
        
        for bar in (self.overall_bar, self.step1_bar, self.step2_bar, self.step3_bar, self.step4_bar):
            bar.set_color(self.accent_blue)
            self.stop_indeterminate(bar)
            self.update_progress(bar, 0)
            
        threading.Thread(target=self.run_pipeline, daemon=True).start()

    def clear_cache_now(self):
        """Standalone action (the OPTIONS card's "Clear Cache Now" button)
        -- wipes the disk cache and any leftover _temp/ scratch folders
        immediately, independent of running a conversion. Runs off the
        main thread since the cache can grow into the tens of GB (see
        cache_utils.py's own module docstring) and a big delete shouldn't
        freeze the window."""
        if not messagebox.askyesno(
                "Clear Cache",
                f"Delete the entire disk cache at:\n{cache_utils.cache_root()}\n\n"
                f"and any leftover temp scratch folders? This can't be undone, but "
                f"everything in it is fully reproducible by re-running a conversion."):
            return
        threading.Thread(target=wipe_cache_and_temp, args=(self.log,), daemon=True).start()

    def _glass_opacity(self):
        try:
            return int(self.glass_opacity_var.get())
        except (tk.TclError, ValueError):
            return 50

    def _max_texture(self):
        try:
            return max(0, int(self.max_texture_var.get()))
        except (tk.TclError, ValueError):
            return 2048

    def run_pipeline(self):
        opts = PipelineOptions(
            pkg_dir=self.pkg_var.get(),
            out_dir=self.out_var.get(),
            msfs_install_dir=self.msfs_var.get(),
            propdefs_dir=self.propdefs_var.get(),
            xp_version=self.xp_version_var.get(),
            clean_run=self.clean_run_var.get(),
            disable_cache=self.disable_cache_var.get(),
            static_doors=self.static_doors_var.get(),
            approximate_substitution=self.approximate_substitution_var.get(),
            scan_terrain_vectors=self.scan_terrain_vectors_var.get(),
            pol_polygons=self.pol_polygons_var.get(),
            prompt_replacements=self.prompt_replacements_var.get(),
            runway_surface="native" if self.native_runways_var.get() else "transparent",
            native_painted_lines=self.native_painted_lines_var.get(),
            flatten_airport=self.flatten_airport_var.get(),
            remove_runway_clutter=self.remove_runway_clutter_var.get(),
            exclusions=self.exclusions_var.get(),
            write_log_file=self.write_log_file_var.get(),
            no_autodgs=self.no_autodgs_var.get(),
            glass_opacity=max(1, min(100, self._glass_opacity())),
            max_texture=self._max_texture(),
        )
        try:
            run_pipeline(opts, _GuiHooks(self))
        except Exception as e:
            self.log(f"Pipeline failed: {e}", "error")
        finally:
            self.root.after(0, lambda: self.start_btn.set_state("normal"))


def _version_text():
    v = app_version()
    return f"v{v}" if v[:1].isdigit() else v


class _GuiHooks(PipelineHooks):
    """Routes pipeline callbacks onto the Tk main thread."""

    def __init__(self, app):
        self.app = app

    def _bar(self, step):
        return getattr(self.app, f"{step}_bar")

    def log(self, text, level="info"):
        self.app.log(text, level)

    def progress(self, step, value, maximum=100):
        self.app.update_progress(self._bar(step), value, maximum)

    def indeterminate(self, step):
        self.app.set_indeterminate(self._bar(step))

    def stop_indeterminate(self, step):
        self.app.stop_indeterminate(self._bar(step))

    def mark_complete(self):
        self.app.mark_overall_complete()

    def prompt_replacements(self, pkg, xplane_root):
        # run_pipeline runs on a worker thread; the Toplevel has to be
        # built on the Tk main thread, so hand it over and wait.
        done = threading.Event()

        def _open():
            try:
                pick_replacements.prompt_modal(self.app.root, pkg, xplane_root)
            except Exception as e:
                self.log(f"Replacement picker failed ({e}) -- continuing without it.", "warning")
            finally:
                done.set()

        self.app.root.after(0, _open)
        if not done.wait(timeout=3600):
            self.log("Replacement picker still open after 60 min -- continuing without waiting.", "warning")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    # "MSFS2XP cli <args>" (or "MSFS2XP.AppImage cli <args>") runs the
    # headless converter from the same packaged build -- see cli.py.
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        import cli
        sys.exit(cli.main(sys.argv[2:]))
    root = tk.Tk()
    app = ModularPythonConverterApp(root)
    root.mainloop()
