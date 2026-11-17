package players;

import core.Types;
import core.actions.Action;
import core.actions.tribeactions.EndTurn;
import core.actions.unitactions.Attack;
import core.actions.unitactions.Move;
import core.actors.City;
import core.actors.units.Unit;
import core.game.Board;
import core.game.FogMemory;
import core.game.GameState;
import core.game.PythonEnv;
import org.json.JSONObject;
import utils.Vector2d;

import java.util.ArrayList;
import java.util.function.Predicate;

/**
 * Puzzles for the combat fog memory (core.game.FogMemory). Run from pol_env/Tribes with the engine on the classpath:
 *   java -cp out:lib/json.jar:<test classes> players.FogMemoryTest levels/phase2_pool_1v1/train/map_2p_010000.csv
 */
public final class FogMemoryTest {

    private static String duelLevel;
    private static int checks = 0;

    private FogMemoryTest() {}

    public static void main(String[] args) {
        if (args.length < 1) throw new IllegalArgumentException("usage: FogMemoryTest <two-tribe level csv>");
        duelLevel = args[0];
        testOffByDefault();
        testWatchedMoveIsFollowedIntoTheFog();
        testUnwatchedMoveLeavesAReachableRegionThatGrows();
        testAttackRevealsTheAttacker();
        testSeenDeathForgetsTheUnit();
        testBlockingShrinksWhereAUnitCanBe();
        testUnseenUnitsFromAnUnfoundCapital();
        testSearchSeesARememberedUnit();
        System.out.println("FogMemoryTest: PASS (" + checks + " checks)");
    }

    // ---- fixtures ----------------------------------------------------------------------------------------------

    /** All plain, no units, both tribes see everything, tribe 1's turn is left to the caller (no opponent agent). */
    private static PythonEnv emptyDuel() {
        PythonEnv env = new PythonEnv();
        env.setOpponentAgent("external", 0);
        env.initFromLevel(duelLevel, 4321L, Types.GAME_MODE.CAPITALS);
        GameState gs = env.getGameState();
        Board b = gs.getBoard();
        for (int i = 0; i < 2; i++) for (Unit u : gs.getUnits(i)) gs.killUnit(u);
        for (int x = 0; x < b.getSize(); x++)
            for (int y = 0; y < b.getSize(); y++)
                if (b.getTerrainAt(x, y) != Types.TERRAIN.CITY) { b.setTerrainAt(x, y, Types.TERRAIN.PLAIN); b.setResourceAt(x, y, null); }
        for (int i = 0; i < 2; i++) for (boolean[] row : gs.getTribe(i).getObsGrid()) java.util.Arrays.fill(row, true);
        return env;
    }

    /** Tribe 0 stops seeing every tile with y > edge (the "clouds" are south of its units). */
    private static void fogSouthOf(GameState gs, int edge) {
        boolean[][] g = gs.getTribe(0).getObsGrid();
        for (int x = 0; x < g.length; x++) for (int y = edge + 1; y < g.length; y++) g[x][y] = false;
    }

    private static Unit place(GameState gs, int tribe, Types.UNIT type, int x, int y) {
        City home = (City) gs.getActor(gs.getTribe(tribe).getCapitalID());
        Unit u = Types.UNIT.createUnit(new Vector2d(x, y), 0, false, -1, tribe, type);
        gs.getBoard().addUnit(home, u);
        u.setStatus(Types.TURN_STATUS.FRESH);
        return u;
    }

    /** A tile whose 5x5 square to the south-east (every tile these puzzles use) holds no city. */
    private static Vector2d openSpot(GameState gs) {
        Board b = gs.getBoard();
        for (int y = 1; y + 4 < b.getSize(); y++)
            for (int x = 1; x + 4 < b.getSize(); x++) {
                boolean ok = true;
                for (int dx = -1; dx <= 4 && ok; dx++)
                    for (int dy = -1; dy <= 4 && ok; dy++)
                        if (b.getTerrainAt(x + dx, y + dy) == Types.TERRAIN.CITY) ok = false;
                if (ok) return new Vector2d(x, y);
            }
        throw new AssertionError("no open spot on this map");
    }

    private static int step(PythonEnv env, Predicate<Action> which, String what) {
        GameState gs = env.getGameState();
        env.recomputePlayerActions(gs.getActiveTribeID());
        ArrayList<Action> acts = gs.getAllAvailableActions();
        for (int i = 0; i < acts.size(); i++) {
            if (which.test(acts.get(i))) { env.stepByIndex(i); return i; }
        }
        throw new AssertionError("no such action: " + what + " in " + acts);
    }

    private static void endTurn(PythonEnv env) {
        step(env, a -> a instanceof EndTurn, "END_TURN");
    }

    private static void move(PythonEnv env, Unit u, int x, int y) {
        step(env, a -> a instanceof Move && ((Move) a).getUnitId() == u.getActorId()
                && ((Move) a).getDestination().x == x && ((Move) a).getDestination().y == y, "move to " + x + "," + y);
    }

    private static void attack(PythonEnv env, Unit u, Unit target) {
        step(env, a -> a instanceof Attack && ((Attack) a).getUnitId() == u.getActorId()
                && ((Attack) a).getTargetId() == target.getActorId(), "attack");
    }

    private static int count(boolean[][] g) {
        int n = 0;
        if (g != null) for (boolean[] row : g) for (boolean v : row) if (v) n++;
        return n;
    }

    // ---- puzzles -----------------------------------------------------------------------------------------------

    private static void testOffByDefault() {
        PythonEnv env = emptyDuel();
        assertTrue(!new JSONObject(env.observationJson()).has("fogMemory"), "no fogMemory block when off");
        assertEq(null, env.getFogMemoryOf(0), "nothing tracked when off");
        env.setFogMemory(true, true);
        JSONObject fm = new JSONObject(env.observationJson()).getJSONObject("fogMemory");
        assertTrue(fm.has("presence") && fm.has("unseen") && fm.has("unseenEstimate"), "fogMemory block when on");
    }

    private static void testWatchedMoveIsFollowedIntoTheFog() {
        PythonEnv env = emptyDuel();
        GameState gs = env.getGameState();
        Vector2d p = openSpot(gs);
        place(gs, 0, Types.UNIT.WARRIOR, p.x, p.y);
        Unit rider = place(gs, 1, Types.UNIT.RIDER, p.x + 3, p.y + 1);
        fogSouthOf(gs, p.y + 1);
        env.setFogMemory(true, true);
        endTurn(env);
        move(env, rider, p.x + 3, p.y + 3);  // into the clouds, in plain sight of where it started
        FogMemory fm = env.getFogMemoryOf(0);
        boolean[][] where = fm.possibleTiles(rider.getActorId());
        assertEq(1, count(where), "followed: exactly one possible tile");
        assertTrue(where[p.x + 3][p.y + 3], "where it stopped");
        assertTrue(!gs.getTribe(0).isVisible(p.x + 3, p.y + 3), "a tile we cannot see");
    }

    private static void testUnwatchedMoveLeavesAReachableRegionThatGrows() {
        PythonEnv env = emptyDuel();
        GameState gs = env.getGameState();
        Vector2d p = openSpot(gs);
        place(gs, 0, Types.UNIT.WARRIOR, p.x, p.y);
        Unit rider = place(gs, 1, Types.UNIT.RIDER, p.x + 3, p.y + 1);
        fogSouthOf(gs, p.y + 1);
        env.setFogMemory(true, false);
        endTurn(env);
        move(env, rider, p.x + 3, p.y + 3);
        boolean[][] region = env.getFogMemoryOf(0).possibleTiles(rider.getActorId());
        int n1 = count(region);
        assertTrue(n1 > 1, "a region, not a tile: " + n1);
        assertTrue(region[p.x + 3][p.y + 3], "the region holds where it really is");
        boolean anyVisible = false;
        for (int x = 0; x < region.length; x++) for (int y = 0; y < region.length; y++)
            if (region[x][y] && gs.getTribe(0).isVisible(x, y)) anyVisible = true;
        assertTrue(!anyVisible, "tiles we can see are ruled out");
        endTurn(env);  // tribe 1 ends
        endTurn(env);  // tribe 0 ends: tribe 1 can move again, so the region grows
        int n2 = count(env.getFogMemoryOf(0).possibleTiles(rider.getActorId()));
        assertTrue(n2 > n1, "the region grows by a turn of movement: " + n1 + " -> " + n2);
    }

    private static void testAttackRevealsTheAttacker() {
        PythonEnv env = emptyDuel();
        GameState gs = env.getGameState();
        Vector2d p = openSpot(gs);
        Unit ours = place(gs, 0, Types.UNIT.DEFENDER, p.x, p.y);
        Unit archer = place(gs, 1, Types.UNIT.ARCHER, p.x, p.y + 2);
        fogSouthOf(gs, p.y + 1);
        env.setFogMemory(true, false);  // even without following moves, who hits you is seen
        assertEq(0, env.getFogMemoryOf(0).hiddenTracks().size(), "nothing known before the attack");
        endTurn(env);
        attack(env, archer, ours);
        boolean[][] where = env.getFogMemoryOf(0).possibleTiles(archer.getActorId());
        assertEq(1, count(where), "the attacker is revealed");
        assertTrue(where[p.x][p.y + 2], "where it shot from");
    }

    private static void testSeenDeathForgetsTheUnit() {
        PythonEnv env = emptyDuel();
        GameState gs = env.getGameState();
        Vector2d p = openSpot(gs);
        Unit ours = place(gs, 0, Types.UNIT.SWORDMAN, p.x, p.y);
        Unit weak = place(gs, 1, Types.UNIT.WARRIOR, p.x + 1, p.y);
        weak.setCurrentHP(1);
        env.setFogMemory(true, true);
        attack(env, ours, weak);
        assertEq(null, gs.getActor(weak.getActorId()), "the warrior dies");
        assertEq(null, env.getFogMemoryOf(0).possibleTiles(weak.getActorId()), "and is forgotten");
        assertEq(0, env.getFogMemoryOf(0).hiddenTracks().size(), "no hidden tracks left");
    }

    private static void testBlockingShrinksWhereAUnitCanBe() {
        int[] sizes = new int[2];
        for (int blocked = 0; blocked < 2; blocked++) {
            PythonEnv env = emptyDuel();
            GameState gs = env.getGameState();
            Vector2d p = openSpot(gs);
            place(gs, 0, Types.UNIT.WARRIOR, p.x, p.y);
            Unit rider = place(gs, 1, Types.UNIT.RIDER, p.x + 3, p.y + 1);
            fogSouthOf(gs, p.y + 1);
            env.setFogMemory(true, true);
            endTurn(env);
            move(env, rider, p.x + 3, p.y + 3);
            endTurn(env);
            if (blocked == 1)  // surround it with our units before its next turn
                for (int dx = -1; dx <= 1; dx++) for (int dy = -1; dy <= 1; dy++)
                    if (dx != 0 || dy != 0) place(gs, 0, Types.UNIT.DEFENDER, p.x + 3 + dx, p.y + 3 + dy);
            endTurn(env);
            sizes[blocked] = count(env.getFogMemoryOf(0).possibleTiles(rider.getActorId()));
        }
        assertTrue(sizes[0] > 1, "free, it can be on many tiles: " + sizes[0]);
        assertEq(1, sizes[1], "surrounded, it cannot have moved");
    }

    private static void testUnseenUnitsFromAnUnfoundCapital() {
        PythonEnv env = emptyDuel();
        GameState gs = env.getGameState();
        Vector2d cap = gs.getActor(gs.getTribe(1).getCapitalID()).getPosition();
        boolean[][] g = gs.getTribe(0).getObsGrid();
        for (int x = 0; x < g.length; x++)
            for (int y = 0; y < g.length; y++)
                if (Math.max(Math.abs(x - cap.x), Math.abs(y - cap.y)) <= 2) g[x][y] = false;
        env.setFogMemory(true, true);
        JSONObject fm = new JSONObject(env.fogMemoryJson(0));
        assertEq(2.0, fm.getDouble("unseenEstimate"), "an unfound level-1 capital supports two units");
        double sum = 0.0;
        boolean onlyFog = true;
        for (int x = 0; x < g.length; x++)
            for (int y = 0; y < g.length; y++) {
                double v = fm.getJSONArray("unseen").getJSONArray(x).getDouble(y);
                sum += v;
                if (v > 0 && g[x][y]) onlyFog = false;
            }
        assertTrue(Math.abs(sum - 2.0) < 1e-2, "the estimate is spread over the fog: " + sum);
        assertTrue(onlyFog, "only on tiles we cannot see");
    }

    private static void testSearchSeesARememberedUnit() {
        PythonEnv env = emptyDuel();
        GameState gs = env.getGameState();
        Vector2d p = openSpot(gs);
        Unit b = place(gs, 0, Types.UNIT.WARRIOR, p.x, p.y);
        b.setCurrentHP(3);
        Unit rider = place(gs, 1, Types.UNIT.RIDER, p.x + 3, p.y + 1);
        fogSouthOf(gs, p.y + 1);
        env.setFogMemory(true, true);
        endTurn(env);
        move(env, rider, p.x + 1, p.y + 2);  // retreats into the clouds, two tiles from our weakened warrior
        endTurn(env);                        // our turn
        assertEq(0, gs.getActiveTribeID(), "our turn");
        assertEq(0.0, env.tacticalDanger(false), "without memory the search sees no threat");
        double danger = env.tacticalDanger(true);
        assertTrue(danger > 0, "with memory our warrior is in danger: " + danger);
    }

    private static void assertTrue(boolean ok, String what) {
        checks++;
        if (!ok) throw new AssertionError(what);
    }

    private static void assertEq(Object expected, Object actual, String what) {
        checks++;
        if (expected == null ? actual != null : !expected.equals(actual))
            throw new AssertionError(what + ": expected " + expected + " but was " + actual);
    }
}
