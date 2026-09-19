--[[
msfs2xp Proximity Animator
===========================
Drives the "proximity" (Z:VisibleRadiusBox) animations produced by the
msfs2xp MSFS-to-X-Plane scenery converter: doors, boom barriers and gates
that MSFS opens/raises whenever the user aircraft gets close. Stock X-Plane
has no built-in dataref for "is the aircraft near this specific placed
object" -- nothing in the sim tracks that -- so those animations need a
real plugin to supply one. That is all this script does.

HOW IT WORKS
------------
The converter writes one manifest per converted airport, at:
    <Custom Scenery>/<airport pack>/plugin_data/msfs2xp_proximity.dat
a plain text file, one line per placed object:
    <dataref-name> <lat> <lon>
where <dataref-name> is a custom dataref (e.g.
"msfs2xp/proximity/SHS_Clutter_Barriers_004_LOD0") that the object's own
.obj file already references in an ANIM_trans/ANIM_rotate block -- X-Plane
just doesn't know what value to give it without something driving it.

On load, this script scans every folder under Custom Scenery for that
manifest, registers each unique dataref it finds (FlyWithLua creates it the
first time a script binds to a path that isn't already a real sim
dataref), and every frame moves each dataref's value toward 1.0 (open) when
the user aircraft is within TRIGGER_RADIUS_M of ANY of that dataref's known
placements, or back toward 0.0 (closed) otherwise -- eased over
TRANSITION_SECONDS instead of snapping, the same way MSFS's own "Lag"
animation parameter smooths these.

INSTALL
-------
Requires FlyWithLua NG (not the old, unmaintained "FlyWithLua Air Manager"
edition -- NG bundles LuaFileSystem, which this script needs to search
Custom Scenery). Drop this file into:
    X-Plane/Resources/plugins/FlyWithLua/Scripts/
and restart X-Plane (or use FlyWithLua's "Reload all Lua script files"
menu item). Check Log.txt for a line starting with "[msfs2xp proximity]"
confirming how many objects/datarefs it found.

SCOPE
-----
Only objects using this exact MSFS trigger shape are covered -- see the
converter's own parse_time_behavior()/_PROXIMITY_RE for what qualifies.
Anything IK/velocity-driven (jetways) or mission-scripted (ground vehicles)
is out of scope for a script this size and is intentionally left static.
--]]

local TRIGGER_RADIUS_M = 30.0      -- distance at which an object starts opening
local TRANSITION_SECONDS = 2.0     -- time to fully open/close once triggered
local RESCAN_ON_START = true       -- scan Custom Scenery once, at script load

local EARTH_M_PER_DEG_LAT = 111320.0

-- {[dataref_name] = { placements = {{lat=.., lon=..}, ...}, value = 0.0 }}
local proximityObjects = {}
local manifestCount = 0
local placementCount = 0

local function log(msg)
    logMsg("[msfs2xp proximity] " .. msg)
end

local function findXPlaneRoot()
    -- SCRIPT_DIRECTORY (a FlyWithLua global) is always
    -- ".../Resources/plugins/FlyWithLua/Scripts/" -- four levels below the
    -- X-Plane install root, regardless of where X-Plane itself is installed.
    -- Guarded rather than assumed: an uncaught error concatenating a nil
    -- global here, at script-load time, is exactly the kind of mistake
    -- that can take down FlyWithLua's shared Lua state for every other
    -- script too -- see the pcall wrapping around every call site below.
    if type(SCRIPT_DIRECTORY) ~= "string" then
        error("SCRIPT_DIRECTORY is not available (unexpected FlyWithLua version?)")
    end
    return SCRIPT_DIRECTORY .. "../../../../"
end

local function parseManifestLine(line)
    if line:sub(1, 1) == "#" or line:match("^%s*$") then
        return nil
    end
    local datarefName, lat, lon = line:match("^(%S+)%s+(%-?[%d%.]+)%s+(%-?[%d%.]+)%s*$")
    if not datarefName then
        return nil
    end
    return datarefName, tonumber(lat), tonumber(lon)
end

local function loadManifest(path)
    local f = io.open(path, "r")
    if not f then
        return
    end
    manifestCount = manifestCount + 1
    for line in f:lines() do
        local datarefName, lat, lon = parseManifestLine(line)
        if datarefName and lat and lon then
            local entry = proximityObjects[datarefName]
            if not entry then
                -- Creates the custom dataref the object's own OBJ8 ANIM_
                -- block is keyed to, via create_dataref_table (the real
                -- FlyWithLua API for minting a NEW custom dataref --
                -- dataref() only binds to one that already exists
                -- elsewhere). Returns a size-1 table written via
                -- datarefTable[0] below, not a plain Lua variable.
                local luaVarName = datarefName:gsub("[^%w_]", "_")
                local datarefTable = create_dataref_table(datarefName, "Float")
                entry = { placements = {}, value = 0.0, luaVar = luaVarName, datarefTable = datarefTable }
                proximityObjects[datarefName] = entry
            end
            table.insert(entry.placements, { lat = lat, lon = lon })
            placementCount = placementCount + 1
        end
    end
    f:close()
end

local function scanCustomScenery()
    -- FlyWithLua NG preloads LuaFileSystem as the global "lfs" table; on
    -- some builds it's only reachable via require("lfs") instead, so try
    -- the global first and fall back rather than assuming either shape.
    local lfs = _G.lfs
    if not lfs then
        local ok
        ok, lfs = pcall(require, "lfs")
        if not ok then
            lfs = nil
        end
    end
    if not lfs then
        log("ERROR: LuaFileSystem ('lfs') is not available -- this needs FlyWithLua NG, " ..
            "not the old FlyWithLua Air Manager edition. Proximity animations will stay static.")
        return
    end

    local customScenery = findXPlaneRoot() .. "Custom Scenery/"
    for entry in lfs.dir(customScenery) do
        if entry ~= "." and entry ~= ".." then
            local manifestPath = customScenery .. entry .. "/plugin_data/msfs2xp_proximity.dat"
            local attr = lfs.attributes(manifestPath)
            if attr and attr.mode == "file" then
                loadManifest(manifestPath)
            end
        end
    end

    local uniqueCount = 0
    for _ in pairs(proximityObjects) do
        uniqueCount = uniqueCount + 1
    end
    log(string.format(
        "loaded %d manifest(s), %d placement(s), %d unique proximity dataref(s). Trigger radius %.0fm.",
        manifestCount, placementCount, uniqueCount, TRIGGER_RADIUS_M))
end

local function nearestDistanceMeters(placements, lat0, lon0)
    local lonScale = EARTH_M_PER_DEG_LAT * math.cos(math.rad(lat0))
    local best = nil
    for _, p in ipairs(placements) do
        local dy = (p.lat - lat0) * EARTH_M_PER_DEG_LAT
        local dx = (p.lon - lon0) * lonScale
        local d = math.sqrt(dx * dx + dy * dy)
        if not best or d < best then
            best = d
        end
    end
    return best or math.huge
end

-- Everything from here down runs every frame, so it can't be allowed to
-- throw either -- an uncaught error inside a do_every_frame callback
-- degrades FlyWithLua's whole frame loop, not just this script. The real
-- work happens in updateProximityUnsafe(); the global entry point below
-- (the only thing do_every_frame ever calls) wraps it in pcall and, on any
-- error, disables itself with a single log line instead of erroring every
-- single frame forever.
local proximityUpdateDisabled = false

local function updateProximityUnsafe()
    if TRANSITION_SECONDS <= 0 then
        return
    end
    if type(LATITUDE) ~= "number" or type(LONGITUDE) ~= "number" then
        return  -- no aircraft position yet (e.g. still at the main menu)
    end
    -- DO_EVERY_FRAME_TIME_SEC is a FlyWithLua built-in global: the last
    -- frame's duration in seconds, refreshed every frame -- used here so
    -- the open/close speed is frame-rate independent. Falls back to a
    -- plausible frame time if it's ever missing rather than erroring the
    -- whole update.
    local period = DO_EVERY_FRAME_TIME_SEC
    if type(period) ~= "number" or period <= 0 then
        period = 1.0 / 30.0
    end
    local step = period / TRANSITION_SECONDS

    for _, entry in pairs(proximityObjects) do
        local target = (nearestDistanceMeters(entry.placements, LATITUDE, LONGITUDE) <= TRIGGER_RADIUS_M) and 1.0 or 0.0
        if entry.value < target then
            entry.value = math.min(target, entry.value + step)
        elseif entry.value > target then
            entry.value = math.max(target, entry.value - step)
        end
        -- entry.datarefTable is the create_dataref_table() handle from
        -- loadManifest() -- writing index 0 pushes the value into the sim,
        -- so every OBJ8 ANIM_ block reading this dataref sees it update.
        entry.datarefTable[0] = entry.value
    end
end

function msfs2xp_update_proximity()
    if proximityUpdateDisabled then
        return
    end
    local ok, err = pcall(updateProximityUnsafe)
    if not ok then
        proximityUpdateDisabled = true
        log("ERROR during per-frame update, disabling further updates so this " ..
            "doesn't spam every frame: " .. tostring(err))
    end
end

if RESCAN_ON_START then
    local ok, err = pcall(scanCustomScenery)
    if not ok then
        log("ERROR during startup scan (proximity animations will stay static): " .. tostring(err))
    end
end

do_every_frame("msfs2xp_update_proximity()")
