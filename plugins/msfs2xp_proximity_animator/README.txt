msfs2xp Proximity Animator (FlyWithLua script)
================================================

WHAT THIS IS FOR
-----------------
The msfs2xp converter turns most MSFS SimObject animations into plain
X-Plane OBJ8 ANIM_ blocks driven by real, always-available X-Plane datarefs
(sim/time/local_time_sec for business-hours doors, a sin-wave dataref for
blinking lights) -- no plugin needed for those.

A handful of objects use MSFS's "Z:VisibleRadiusBox" trigger instead: doors,
boom barriers and gates that open only when the user aircraft gets close.
Nothing in stock X-Plane tracks "is the aircraft near this specific placed
object", so those few objects genuinely need a plugin. This script is that
plugin -- nothing more. Every other animation in the converted scenery
already works with no plugin installed at all; this only affects the small
number of objects that were proximity-triggered in the original MSFS
package (you'll see them listed by count in the converter's own log, under
"Wrote proximity-animation manifest").

Without this script installed, those specific objects still render
correctly -- they just stay in their default (closed) pose instead of
opening as you approach.

REQUIREMENTS
------------
FlyWithLua NG for X-Plane 11 or 12 (the actively maintained edition -- not
the old, retired "FlyWithLua Air Manager"). Get it from the X-Plane.org
forums or flightsim marketplaces you already use for other X-Plane addons.
NG is required specifically because it bundles LuaFileSystem, which this
script uses to search your Custom Scenery folder for converted airports.

INSTALL
-------
1. Install FlyWithLua NG for X-Plane if you haven't already.
2. Copy msfs2xp_proximity_animator.lua into:
       X-Plane/Resources/plugins/FlyWithLua/Scripts/
3. Start X-Plane (or, if it's already running, use the FlyWithLua menu ->
   "Reload all Lua script files").
4. Open Log.txt (in your X-Plane root folder) and look for a line starting
   with "[msfs2xp proximity]" -- it reports how many manifests, placements
   and unique datarefs it found. If it says 0 of everything, either no
   msfs2xp-converted airport with proximity objects is installed, or the
   scan failed (the same log line explains why).

This script re-scans Custom Scenery once, when it loads. If you convert or
install a new airport afterward, reload FlyWithLua's scripts (or restart
X-Plane) to pick it up.

TUNING
------
Two constants at the top of the .lua file:
    TRIGGER_RADIUS_M    -- distance (meters) at which an object starts
                            opening. Default 30.
    TRANSITION_SECONDS  -- how long a full open/close takes once triggered.
                            Default 2.

SCOPE / LIMITATIONS
--------------------
- Only the exact MSFS trigger shape "(Z:VisibleRadiusBox, Number) 1 == if{
  N } els{ 0 }" is covered -- this is what the converter itself detects and
  emits a manifest entry for. Anything IK/velocity-driven (jetways) or
  mission-scripted (ground vehicles following a route) is a fundamentally
  different, much larger problem and is intentionally out of scope here.
- One shared trigger radius/transition time for every object in this
  script, rather than each object's own (MSFS doesn't expose that value in
  a way the converter can read reliably either).
- If two placements share the same source model, they share the same
  dataref and therefore move together based on whichever one the aircraft
  is closer to -- fine for the common case (each door/barrier in a real
  MSFS package is normally its own distinct model already), but a model
  reused verbatim at several different locations would open all of its
  copies together if the aircraft is near just one.
