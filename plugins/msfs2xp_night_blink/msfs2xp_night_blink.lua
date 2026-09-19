--[[
msfs2xp Night Blink
====================
Drives the "msfs2xp/night_blink" custom dataref that the msfs2xp converter
wires up for periodic-blink objects (beacons, lamp bulbs, etc -- anything
whose MSFS behavior XML matched the "blink" trigger shape, see the
converter's own parse_time_behavior()/_BLINK_RE). Those objects use
X-Plane's ATTR_light_level to blend in a self-lit TEXTURE_LIT texture, and
X-Plane's own OBJ8 spec is explicit that ATTR_light_level "overrides the
sim's decision about object lighting" -- i.e. it REPLACES the automatic
day/night glow entirely for whatever it's applied to, it doesn't add to it.
So the dataref driving it has to carry BOTH the blink shape AND real
day/night gating itself, in one single value, or the object blinks exactly
the same at noon as at midnight (day/night gating silently lost the moment
ATTR_light_level is used at all).

HOW IT WORKS
------------
Every frame, this script computes:
    night_blink = percent_lights_on * blink_wave
and pushes it into the "msfs2xp/night_blink" dataref, where:
    - percent_lights_on is X-Plane's own stock
      "sim/graphics/scenery/percent_lights_on" dataref (0.0 in full
      daylight, ramping up to 1.0 at night, with X-Plane's own built-in
      dusk/dawn transition and hysteresis -- not reimplemented here,
      just reused).
    - blink_wave is a smooth 0..1 periodic wave.
The converter calibrates ATTR_light_level's "on" transition between 0.3 and
0.7 -- during full daylight night_blink is pinned at/near 0 regardless of
blink_wave (below the 0.3 floor, reads as fully off); during full night it
sweeps through the same 0.3-0.7 window on every blink cycle, same as
before; during dusk/dawn it fades in/out gradually rather than snapping on,
which reads as a natural transition rather than a bug.

This is a single SHARED dataref, not one per placement/object like the
companion "msfs2xp Proximity Animator" script -- unlike proximity (which
depends on the aircraft's distance to a SPECIFIC placement), "is it night
and what's the current blink phase" is the same answer everywhere in the
sim at any given moment, so every blinking object in every converted
airport reads the exact same value with no per-object bookkeeping needed.

INSTALL
-------
Requires FlyWithLua (NG or Air Manager both work -- this script, unlike the
companion proximity one, doesn't need LuaFileSystem). Drop this file into:
    X-Plane/Resources/plugins/FlyWithLua/Scripts/
and restart X-Plane (or use FlyWithLua's "Reload all Lua script files"
menu item). Check Log.txt for a line starting with "[msfs2xp night blink]"
confirming it loaded.

Without this script installed, blink objects fall back to X-Plane's
default handling of an out-of-range/undefined dataref (typically reading
as 0), which -- being below the 0.3 calibration floor -- reads as
"steady off" rather than blinking. They still render correctly otherwise;
only the blink animation itself is affected.
--]]

local BLINK_PERIOD_SECONDS = 2.0   -- full on-off-on cycle length

local NIGHT_BLINK_DATAREF = "msfs2xp/night_blink"
local BLINK_ALWAYS_DATAREF = "msfs2xp/blink_always"

local function log(msg)
    logMsg("[msfs2xp night blink] " .. msg)
end

-- Registers (and, since no real sim dataref of this name exists, CREATES
-- as a writable float) the custom dataref every "blink" object's own
-- ATTR_light_level is keyed to.
local msfs2xp_night_blink_out = create_dataref_table(NIGHT_BLINK_DATAREF, "Float")

-- Same blink_wave, but with no percent_lights_on gating -- for
-- LIGHT_SPILL_CUSTOM point lights whose ASOBO_macro_light data marks them
-- as flashing but NOT day/night-gated (day_night_cycle=false), e.g. a
-- rotating beacon: it flashes day and night alike in reality, so gating it
-- off in daylight the way the night-only case needs would be wrong.
local msfs2xp_blink_always_out = create_dataref_table(BLINK_ALWAYS_DATAREF, "Float")

-- Reads X-Plane's own stock day/night dataref -- this is what supplies
-- real, sim-managed dusk/dawn gating; nothing here reimplements sun-angle
-- math itself.
dataref("msfs2xp_percent_lights_on", "sim/graphics/scenery/percent_lights_on")

local phase = 0.0
local updateDisabled = false

local function updateNightBlinkUnsafe()
    local period = DO_EVERY_FRAME_TIME_SEC
    if type(period) ~= "number" or period <= 0 then
        period = 1.0 / 30.0
    end

    phase = phase + period
    if phase >= BLINK_PERIOD_SECONDS then
        phase = phase % BLINK_PERIOD_SECONDS
    end

    -- Smooth 0..1 wave, one full cycle per BLINK_PERIOD_SECONDS.
    local blink_wave = (math.sin(2 * math.pi * phase / BLINK_PERIOD_SECONDS) + 1.0) / 2.0

    local night = msfs2xp_percent_lights_on
    if type(night) ~= "number" then
        night = 0.0
    end
    night = math.max(0.0, math.min(1.0, night))

    msfs2xp_night_blink_out[0] = night * blink_wave
    msfs2xp_blink_always_out[0] = blink_wave
end

-- Runs every frame, so it can't be allowed to throw -- an uncaught error
-- inside a do_every_frame callback degrades FlyWithLua's whole frame loop,
-- not just this script. On any error, disable further updates with a
-- single log line instead of erroring every frame forever.
function msfs2xp_update_night_blink()
    if updateDisabled then
        return
    end
    local ok, err = pcall(updateNightBlinkUnsafe)
    if not ok then
        updateDisabled = true
        log("ERROR during per-frame update, disabling further updates so this " ..
            "doesn't spam every frame: " .. tostring(err))
    end
end

log(string.format(
    "loaded -- driving \"%s\" (night-gated) and \"%s\" (ungated) from a %.1fs blink period.",
    NIGHT_BLINK_DATAREF, BLINK_ALWAYS_DATAREF, BLINK_PERIOD_SECONDS))

do_every_frame("msfs2xp_update_night_blink()")
