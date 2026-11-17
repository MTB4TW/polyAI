package players;

import core.Types;
import core.actions.Action;
import core.actions.unitactions.Attack;
import core.actions.unitactions.Capture;
import core.actors.City;
import core.actors.Tribe;
import core.actors.units.Unit;
import core.game.Board;
import core.game.GameState;
import core.game.PythonEnv;
import players.heuristics.CombatTactics;
import utils.Vector2d;

import java.util.ArrayList;

/**
 * Puzzles for the combat tactical search (players.heuristics.CombatTactics): attack order, taking a capital, and
 * blocking an enemy from reaching a besieging unit. Run from pol_env/Tribes with the engine on the classpath:
 *   java -cp out:lib/json.jar:<test classes> players.CombatTacticsTest levels/phase2_pool_1v1/train/map_2p_010000.csv
 */
public final class CombatTacticsTest {

    private static String duelLevel;
    private static int checks = 0;

    private CombatTacticsTest() {}

    public static void main(String[] args) {
        if (args.length < 1) throw new IllegalArgumentException("usage: CombatTacticsTest <two-tribe level csv>");
        duelLevel = args[0];
        testAttackOrder();
        testTakesTheCapital();
        testBlockingRemovesTheThreat();
        testNoTacticalActionMeansNoSuggestion();
        System.out.println("CombatTacticsTest: PASS (" + checks + " checks)");
    }

    /** A cleared board: all plain, no units, both tribes see everything, tribe 0 to move. */
    private static PythonEnv emptyDuel() {
        PythonEnv env = new PythonEnv();
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

    private static Unit place(GameState gs, int tribe, Types.UNIT type, int x, int y) {
        City home = (City) gs.getActor(gs.getTribe(tribe).getCapitalID());
        Unit u = Types.UNIT.createUnit(new Vector2d(x, y), 0, false, -1, tribe, type);
        gs.getBoard().addUnit(home, u);
        u.setStatus(Types.TURN_STATUS.FRESH);
        return u;
    }

    /** A free plain tile at distance >= 3 from both capitals, with room around it. */
    private static Vector2d openSpot(GameState gs) {
        Board b = gs.getBoard();
        for (int x = 1; x < b.getSize() - 1; x++)
            for (int y = 2; y < b.getSize() - 2; y++) {
                Vector2d p = new Vector2d(x, y);
                boolean ok = true;
                for (int i = 0; i < 2; i++) {
                    Vector2d c = gs.getActor(gs.getTribe(i).getCapitalID()).getPosition();
                    if (Vector2d.chebychevDistance(p, c) < 4) ok = false;
                }
                if (ok) return p;
            }
        throw new AssertionError("no open spot on this map");
    }

    private static int choose(PythonEnv env, int depth, int beam) {
        env.recomputePlayerActions(0);
        GameState gs = env.getGameState();
        ArrayList<Action> acts = gs.getAllAvailableActions();
        int[] all = new int[acts.size()];
        for (int i = 0; i < all.length; i++) all[i] = i;
        return env.tacticalChooseIndex(null, all, depth, beam, 3, 7L);
    }

    private static void testAttackOrder() {
        // Enemy warrior at full health; our archer two tiles away and our warrior next to it. Either order kills,
        // but the warrior attacking first takes retaliation; the archer first (out of reach) is the better line.
        PythonEnv env = emptyDuel();
        GameState gs = env.getGameState();
        Vector2d p = openSpot(gs);
        Unit enemy = place(gs, 1, Types.UNIT.WARRIOR, p.x, p.y);
        Unit archer = place(gs, 0, Types.UNIT.ARCHER, p.x, p.y - 2);
        Unit warrior = place(gs, 0, Types.UNIT.WARRIOR, p.x - 1, p.y - 1);
        int idx = choose(env, 3, 8);
        Action a = gs.getAllAvailableActions().get(idx);
        assertTrue(a instanceof Attack, "search attacks first: " + a);
        assertEq(archer.getActorId(), ((Attack) a).getUnitId(), "the archer attacks first");
        assertEq(enemy.getActorId(), ((Attack) a).getTargetId(), "at the enemy warrior");
        // Follow the search for the rest of the turn: the enemy dies and our warrior is untouched.
        for (int step = 0; step < 4 && gs.getActor(enemy.getActorId()) != null; step++) {
            int i = choose(env, 3, 8);
            if (i < 0) break;
            env.stepByIndex(i);
        }
        assertEq(null, gs.getActor(enemy.getActorId()), "the enemy warrior is killed this turn");
        assertEq(warrior.getMaxHP(), warrior.getCurrentHP(), "our warrior takes no retaliation");
    }

    private static void testTakesTheCapital() {
        PythonEnv env = emptyDuel();
        GameState gs = env.getGameState();
        Vector2d cap = gs.getActor(gs.getTribe(1).getCapitalID()).getPosition();
        place(gs, 0, Types.UNIT.WARRIOR, cap.x, cap.y);
        int idx = choose(env, 2, 8);
        Action a = gs.getAllAvailableActions().get(idx);
        assertTrue(a instanceof Capture, "search captures the enemy capital: " + a);
        env.stepByIndex(idx);
        assertEq(0, gs.getActor(gs.getTribe(1).getCapitalID()).getTribeId(), "the enemy capital is ours");
    }

    private static void testBlockingRemovesTheThreat() {
        // Our weakened unit B; an enemy warrior two tiles away can step next to it and kill it next turn. With our
        // units on the three tiles it could step to, it cannot reach B: the danger term must vanish.
        PythonEnv env = emptyDuel();
        GameState gs = env.getGameState();
        Vector2d p = openSpot(gs);
        Unit b = place(gs, 0, Types.UNIT.WARRIOR, p.x, p.y);
        b.setCurrentHP(3);
        place(gs, 1, Types.UNIT.WARRIOR, p.x, p.y + 2);
        CombatTactics t = new CombatTactics(0, null, 3);
        double threatened = t.quickValue(gs.copy()) - t.value(gs.copy());
        assertTrue(threatened > 0, "the weakened unit is in danger: " + threatened);
        for (int dx = -1; dx <= 1; dx++) place(gs, 0, Types.UNIT.DEFENDER, p.x + dx, p.y + 1);
        double after = t.quickValue(gs.copy()) - t.value(gs.copy());
        assertEq(0.0, after, "blocked: no unit of ours can be killed next turn");
    }

    private static void testNoTacticalActionMeansNoSuggestion() {
        PythonEnv env = emptyDuel();
        GameState gs = env.getGameState();
        Vector2d p = openSpot(gs);
        place(gs, 0, Types.UNIT.WARRIOR, p.x, p.y);  // far from every enemy actor
        assertEq(-1, choose(env, 3, 8), "nothing tactical: leave the decision to the policy");
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
