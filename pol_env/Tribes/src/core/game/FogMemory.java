package core.game;

import core.Types;
import core.actions.Action;
import core.actions.unitactions.Attack;
import core.actions.unitactions.Move;
import core.actors.Actor;
import core.actors.City;
import core.actors.Tribe;
import core.actors.units.Unit;
import org.json.JSONArray;
import org.json.JSONObject;
import utils.Vector2d;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * Combat only (opt-in, POLYVISION_FOG_MEMORY=1): what one tribe (the observer) can remember and infer about enemy
 * units it cannot currently see. Fog in this engine (as in Polytopia) is unexplored tiles; units on explored tiles
 * are always visible, so the memory is about enemy units inside unexplored tiles (and hidden Cloaks).
 *
 * Only information the observer legitimately has is used:
 * - Tracked units: every enemy unit the observer has seen. While visible, its tile is known. When it leaves view,
 *   its possible tiles are those it could reach from where it was last seen; at the start of each of its tribe's
 *   turns they grow by one turn of movement, and every tile the observer can see (without that unit on it) is
 *   ruled out. A unit whose possible tiles run out is dropped, as is one the observer saw die.
 * - Units we watch: an enemy unit that attacks one of the observer's units is revealed where it stands. With
 *   seeMoves on (POLYVISION_FOG_MEMORY_SEE_MOVES, default on), a move the observer watched start (the unit was
 *   visible or had just attacked) is followed to its destination even if that is in the fog: a Rider that hits
 *   and retreats into the clouds is remembered where it stopped (Polytopia shows such moves).
 * - Unseen units: an estimate of how many enemy units the observer has never seen, from the unit capacity
 *   (city level + 1) of the enemy cities it knows about (an unseen capital counts as level 1), spread over the
 *   fog by distance to those cities.
 *
 * Movement is estimated from the observer's knowledge: unknown (fogged) tiles are passable, known water blocks
 * land units, the observer's units block their tile, and tiles next to them stop movement (zone of control), so
 * blocking with our units shrinks where an enemy unit can be.
 */
public final class FogMemory {

    /** Tracks unseen this many of their tribe's turns are dropped (the unit may have died or embarked). */
    public static final int MAX_TURNS_UNSEEN = 6;

    static final class Track {
        final int id;
        final int tribe;
        Types.UNIT type;
        int hp, mov, atk, range;
        int lastX, lastY, lastSeenTick;
        int turnsUnseen;
        boolean visible;      // visible at the last observation
        boolean watched;      // revealed by an attack this turn: its next move is followed
        boolean[][] possible; // tiles it can be on (only its own tile while visible)

        Track(int id, int tribe) { this.id = id; this.tribe = tribe; }
    }

    private final int observer;
    private final boolean seeMoves;
    private final Map<Integer, Track> tracks = new HashMap<>();
    private int pendingFollow = -1;
    private int lastActiveTribe = -1;

    public FogMemory(int observer, boolean seeMoves) {
        this.observer = observer;
        this.seeMoves = seeMoves;
    }

    public int getObserver() { return observer; }

    // ---- updates ------------------------------------------------------------------------------------------------

    /** Start of a game: forget everything and look at the board. */
    public void reset(GameState gs) {
        tracks.clear();
        pendingFollow = -1;
        lastActiveTribe = gs.getActiveTribeID();
        observe(gs);
    }

    /** Before any action is applied (either tribe's). Notes enemy attacks on us and moves we watch start. */
    public void beforeAction(GameState gs, Action a) {
        pendingFollow = -1;
        if (a instanceof Attack) {
            Attack at = (Attack) a;
            Actor att = gs.getActor(at.getUnitId());
            Actor tgt = gs.getActor(at.getTargetId());
            if (att instanceof Unit && tgt != null && att.getTribeId() != observer && tgt.getTribeId() == observer) {
                Track t = see(gs, (Unit) att);  // you see who hits you
                t.watched = true;
            }
        } else if (a instanceof Move && seeMoves) {
            Actor u = gs.getActor(((Move) a).getUnitId());
            if (u instanceof Unit && u.getTribeId() != observer) {
                Track t = tracks.get(u.getActorId());
                if (visibleTo(gs, (Unit) u) || (t != null && t.watched)) pendingFollow = u.getActorId();
            }
        }
    }

    /** After any action is applied. */
    public void afterAction(GameState gs) {
        int active = gs.getActiveTribeID();
        if (active != lastActiveTribe) {
            for (Track t : tracks.values()) t.watched = false;
            if (active != observer) startTurnOf(gs, active);
            lastActiveTribe = active;
        }
        observe(gs);
        if (pendingFollow >= 0) {
            Actor u = gs.getActor(pendingFollow);
            if (u instanceof Unit && u.getTribeId() != observer) {
                Track t = see(gs, (Unit) u);  // followed to where it stopped, even inside the fog
                t.watched = true;
            }
            pendingFollow = -1;
        }
    }

    /** The observer's view of the current state: visible enemy units pin their tracks, seen tiles are ruled out. */
    void observe(GameState gs) {
        Board b = gs.getBoard();
        Tribe me = gs.getTribe(observer);
        for (Actor a : b.getActors()) {
            if (a instanceof Unit && a.getTribeId() != observer && visibleTo(gs, (Unit) a)) see(gs, (Unit) a);
        }
        Iterator<Track> it = tracks.values().iterator();
        while (it.hasNext()) {
            Track t = it.next();
            Actor a = gs.getActor(t.id);
            boolean nowVisible = a instanceof Unit && a.getTribeId() == t.tribe && visibleTo(gs, (Unit) a);
            if (nowVisible) continue;
            if (a != null && a.getTribeId() == observer) { it.remove(); continue; }  // converted to our side
            if (a == null && t.watched) { it.remove(); continue; }  // a unit we were watching died
            if (t.visible) {
                t.visible = false;
                if (a == null && (t.watched || me.isVisible(t.lastX, t.lastY))) { it.remove(); continue; }  // we saw it die
                // It left our view: it moved into the fog (or became hidden) from its last tile.
                t.possible = reach(gs, single(b.getSize(), t.lastX, t.lastY), t, true);
            }
            int n = 0;
            for (int x = 0; x < t.possible.length; x++)
                for (int y = 0; y < t.possible.length; y++) {
                    if (t.possible[x][y] && me.isVisible(x, y) && !hiddenCloakCould(gs, t, x, y)) t.possible[x][y] = false;
                    if (t.possible[x][y]) n++;
                }
            if (n == 0) it.remove();
        }
    }

    private void startTurnOf(GameState gs, int tribe) {
        Iterator<Track> it = tracks.values().iterator();
        while (it.hasNext()) {
            Track t = it.next();
            if (t.tribe != tribe || t.visible) continue;
            if (++t.turnsUnseen > MAX_TURNS_UNSEEN) { it.remove(); continue; }
            t.possible = reach(gs, t.possible, t, false);
        }
    }

    private Track see(GameState gs, Unit u) {
        Track t = tracks.get(u.getActorId());
        if (t == null) { t = new Track(u.getActorId(), u.getTribeId()); tracks.put(t.id, t); }
        t.type = u.getType();
        t.hp = u.getCurrentHP();
        t.mov = u.MOV;
        t.atk = u.ATK;
        t.range = u.RANGE;
        t.lastX = u.getPosition().x;
        t.lastY = u.getPosition().y;
        t.lastSeenTick = gs.getTick();
        t.turnsUnseen = 0;
        t.visible = visibleTo(gs, u);
        t.possible = single(gs.getBoard().getSize(), t.lastX, t.lastY);
        return t;
    }

    // ---- what the observer knows -------------------------------------------------------------------------------

    private boolean visibleTo(GameState gs, Unit u) {
        Vector2d p = u.getPosition();
        if (!gs.getTribe(observer).isVisible(p.x, p.y)) return false;
        return !(u.isHidden() && !gs.getBoard().getRelations().isAtPeace(u.getTribeId(), observer));
    }

    /** A tracked Cloak may stand hidden on a tile we see: never rule a tile out for it on sight alone. */
    private static boolean hiddenCloakCould(GameState gs, Track t, int x, int y) {
        return t.type == Types.UNIT.CLOAK && gs.getBoard().getUnitAt(x, y) == null;
    }

    private static boolean[][] single(int size, int x, int y) {
        boolean[][] g = new boolean[size][size];
        g[x][y] = true;
        return g;
    }

    /**
     * Tiles reachable from `from` within one turn of movement, by the observer's knowledge (see the class comment).
     * With fogOnly, only tiles the observer cannot see are kept (a unit that just left our view).
     */
    private boolean[][] reach(GameState gs, boolean[][] from, Track t, boolean fogOnly) {
        Board b = gs.getBoard();
        Tribe me = gs.getTribe(observer);
        int size = b.getSize();
        boolean water = t.type != null && t.type.isWaterUnit();
        boolean[][] ours = new boolean[size][size];
        boolean[][] zoc = new boolean[size][size];
        for (Actor a : b.getActors()) {
            if (!(a instanceof Unit) || a.getTribeId() != observer) continue;
            Vector2d p = a.getPosition();
            ours[p.x][p.y] = true;
            for (int dx = -1; dx <= 1; dx++)
                for (int dy = -1; dy <= 1; dy++) {
                    int x = p.x + dx, y = p.y + dy;
                    if (x >= 0 && y >= 0 && x < size && y < size) zoc[x][y] = true;
                }
        }
        int[][] dist = new int[size][size];
        for (int[] row : dist) java.util.Arrays.fill(row, Integer.MAX_VALUE);
        ArrayList<int[]> frontier = new ArrayList<>();
        for (int x = 0; x < size; x++)
            for (int y = 0; y < size; y++)
                if (from[x][y]) { dist[x][y] = 0; frontier.add(new int[]{x, y}); }
        int steps = Math.max(1, t.mov);
        for (int d = 1; d <= steps; d++) {
            ArrayList<int[]> next = new ArrayList<>();
            for (int[] c : frontier) {
                if (dist[c[0]][c[1]] > 0 && zoc[c[0]][c[1]]) continue;  // stopped next to one of our units
                for (int dx = -1; dx <= 1; dx++)
                    for (int dy = -1; dy <= 1; dy++) {
                        int x = c[0] + dx, y = c[1] + dy;
                        if ((dx == 0 && dy == 0) || x < 0 || y < 0 || x >= size || y >= size) continue;
                        if (dist[x][y] != Integer.MAX_VALUE || ours[x][y]) continue;
                        if (me.isVisible(x, y)) {
                            Types.TERRAIN ter = b.getTerrainAt(x, y);
                            boolean isWater = ter == Types.TERRAIN.SHALLOW_WATER || ter == Types.TERRAIN.DEEP_WATER;
                            if (isWater != water && ter != Types.TERRAIN.CITY) continue;
                        }
                        dist[x][y] = d;
                        next.add(new int[]{x, y});
                    }
            }
            frontier = next;
        }
        boolean[][] out = new boolean[size][size];
        for (int x = 0; x < size; x++)
            for (int y = 0; y < size; y++)
                out[x][y] = dist[x][y] != Integer.MAX_VALUE && !(fogOnly && me.isVisible(x, y));
        return out;
    }

    // ---- read-out ----------------------------------------------------------------------------------------------

    /** Enemy units the observer remembers but cannot see now, with their possible tiles. */
    public List<int[]> hiddenTracks() {
        List<int[]> out = new ArrayList<>();
        for (Track t : tracks.values()) if (!t.visible) out.add(new int[]{t.id, t.tribe, t.type.getKey(), t.hp, t.lastX, t.lastY});
        return out;
    }

    /** Possible tiles of a hidden track, or null. */
    public boolean[][] possibleTiles(int unitId) {
        Track t = tracks.get(unitId);
        return t == null || t.visible ? null : t.possible;
    }

    /**
     * The belief as JSON for the observation: per-tile grids [x][y] (as the board arrays) and scalars.
     * presence: expected number of remembered hidden enemy units on the tile (each spread evenly over its tiles);
     * attack: the same weighted by attack / 5; lastSeen: 1 / (1 + turns since seen) on each hidden unit's last
     * tile; unseen: expected number of never-seen enemy units on the tile.
     */
    public JSONObject toJson(GameState gs) {
        Board b = gs.getBoard();
        Tribe me = gs.getTribe(observer);
        int size = b.getSize();
        double[][] presence = new double[size][size], attack = new double[size][size];
        double[][] lastSeen = new double[size][size], unseen = new double[size][size];
        int hidden = 0;
        for (Track t : tracks.values()) {
            if (t.visible) continue;
            hidden++;
            int n = 0;
            for (boolean[] row : t.possible) for (boolean v : row) if (v) n++;
            if (n == 0) continue;
            for (int x = 0; x < size; x++)
                for (int y = 0; y < size; y++)
                    if (t.possible[x][y]) { presence[x][y] += 1.0 / n; attack[x][y] += t.atk / 5.0 / n; }
            int turns = Math.max(0, gs.getTick() - t.lastSeenTick);
            lastSeen[t.lastX][t.lastY] = Math.max(lastSeen[t.lastX][t.lastY], 1.0 / (1 + turns));
        }

        // Never-seen units: capacity of the enemy cities we know (an unseen enemy capital counts as level 1).
        int visibleEnemy = 0, capacity = 0;
        List<Vector2d> enemyCities = new ArrayList<>();
        boolean[] capitalSeen = new boolean[gs.getTribes().length];
        for (Actor a : b.getActors()) {
            if (a.getTribeId() == observer || a.getTribeId() < 0) continue;
            if (a instanceof Unit && visibleTo(gs, (Unit) a)) visibleEnemy++;
            if (a instanceof City && me.isVisible(a.getPosition().x, a.getPosition().y)) {
                City c = (City) a;
                capacity += c.getLevel() + 1;
                enemyCities.add(c.getPosition());
                if (c.isCapital() && c.getTribeId() >= 0 && c.getTribeId() < capitalSeen.length) capitalSeen[c.getTribeId()] = true;
            }
        }
        for (Tribe t : gs.getTribes()) {
            if (t.getTribeId() == observer || capitalSeen[t.getTribeId()] || t.getCapitalID() < 0) continue;
            Actor cap = gs.getActor(t.getCapitalID());
            if (cap != null && cap.getTribeId() == t.getTribeId()) capacity += 2;  // a capital we have not found
        }
        double estimate = Math.max(0, capacity - visibleEnemy - hidden);
        double total = 0.0;
        for (int x = 0; x < size; x++)
            for (int y = 0; y < size; y++) {
                if (me.isVisible(x, y)) continue;
                double w = 1.0;
                if (!enemyCities.isEmpty()) {
                    int d = Integer.MAX_VALUE;
                    for (Vector2d c : enemyCities) d = Math.min(d, Math.max(Math.abs(c.x - x), Math.abs(c.y - y)));
                    w = Math.exp(-d / 2.0);
                }
                unseen[x][y] = w;
                total += w;
            }
        if (total > 0) for (double[] row : unseen) for (int y = 0; y < size; y++) row[y] *= estimate / total;

        JSONObject out = new JSONObject();
        out.put("presence", grid(presence));
        out.put("attack", grid(attack));
        out.put("lastSeen", grid(lastSeen));
        out.put("unseen", grid(unseen));
        out.put("trackedHidden", hidden);
        out.put("unseenEstimate", estimate);
        out.put("knownEnemyCapacity", capacity);
        return out;
    }

    private static JSONArray grid(double[][] g) {
        JSONArray out = new JSONArray();
        for (double[] row : g) {
            JSONArray r = new JSONArray();
            for (double v : row) r.put(Math.round(v * 10000.0) / 10000.0);
            out.put(r);
        }
        return out;
    }
}
