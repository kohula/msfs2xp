msfs2xp Night Blink (FlyWithLua script)
=========================================

WHAT THIS IS FOR
-----------------
The msfs2xp converter turns periodic-blink MSFS objects (beacons, lamp
bulbs -- anything matching the "blink" trigger shape in its behavior XML)
into an X-Plane OBJ8 ATTR_light_level block. X-Plane's own OBJ8 spec is
explicit that ATTR_light_level "overrides the sim's decision about object
lighting" -- it REPLACES the automatic day/night glow entirely for
whatever it's applied to, rather than adding to it. So the dataref driving
it has to carry the blink pattern AND real day/night gating together, in
one value.

This script computes that one value every frame -- a periodic blink wave,
multiplied by X-Plane's own stock day/night dataref
(sim/graphics/scenery/percent_lights_on) -- and drives it into the custom
"msfs2xp/night_blink" dataref every converted blink object's own
ATTR_light_level reads. Without this script, that dataref reads as
undefined/0, which is below the converter's calibration floor -- objects
render correctly otherwise, they just sit steady-off instead of blinking.

The converter also emits real point lights (taxiway edge lights, apron
floodlights, obstruction/beacon lights -- read from each object's
ASOBO_macro_light data) as X-Plane LIGHT_SPILL_CUSTOM commands. Those take
their own dataref straight off macro_light's day_night_cycle/
flash_frequency fields: a plain night-only light uses X-Plane's stock
percent_lights_on directly (no plugin needed at all), a night-only
flashing light reuses this script's "msfs2xp/night_blink", and an
always-on flashing light (e.g. a rotating beacon, which flashes in
daylight too, not just at night) uses a second dataref this same script
drives: "msfs2xp/blink_always" -- the same blink wave, without the
day/night multiply.

Unlike the companion "msfs2xp Proximity Animator" script, this one drives
shared SINGLE datarefs, not one per placement -- "is it night and what's
the current blink phase" is the same answer everywhere in the sim at any
moment, so no per-airport scan or manifest is needed at all.

REQUIREMENTS
------------
Any FlyWithLua build (NG or the older Air Manager edition both work --
this script doesn't need LuaFileSystem, unlike the proximity one).

INSTALL
-------
1. Install FlyWithLua for X-Plane if you haven't already.
2. Copy msfs2xp_night_blink.lua into:
       X-Plane/Resources/plugins/FlyWithLua/Scripts/
3. Start X-Plane (or, if it's already running, use the FlyWithLua menu ->
   "Reload all Lua script files").
4. Open Log.txt (in your X-Plane root folder) and look for a line starting
   with "[msfs2xp night blink]" confirming it loaded.

TUNING
------
One constant at the top of the .lua file:
    BLINK_PERIOD_SECONDS -- full on-off-on cycle length. Default 2.0.
    This is an approximation of MSFS's own zulu-time-modulo blink formula
    (not exposed as any X-Plane dataref), not an exact match -- it's
    picked to visually read as "blinking," not to reproduce the original
    timing precisely.

SCOPE / LIMITATIONS
--------------------
- One shared blink period for every object using this script, same
  reasoning as the proximity script's shared trigger radius.
- Relies on X-Plane's own percent_lights_on for day/night gating rather
  than reimplementing sun-angle math -- inherits whatever dusk/dawn
  transition and hysteresis X-Plane itself uses for that dataref.
