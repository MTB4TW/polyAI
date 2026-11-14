package players.heuristics;

import core.Types;
import core.actions.Action;
import core.actions.unitactions.command.AttackCommand;
import core.actors.Actor;
import core.actors.City;
import core.actors.units.Unit;
import core.game.Board;
import core.game.GameState;
import utils.Pair;
import utils.Vector2d;
import utils.graph.PathNode;
import utils.graph.Pathfinder;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Within-turn tactical search for combat (Phase 2): which attacks, captures, infiltrations and moves near the enemy
 * to make this turn, and in what order. Used as a teacher (PythonEnv.tacticalChooseIndex) and as search at play
 * time (tools/evaluate_combat.py --policy ppo_search).
 *
 * Search: a beam search over sequences of tactical actions of the active tribe, applied to copies of the state
 * (the caller passes a copy with the tribe's fog, so the search only knows what the player sees). Combat damage in
 * the engine is deterministic, so kills and attack orders are computed exactly. Economy actions are left to the
 * policy. The search returns the first action of the best sequence only if that sequence improves on stopping now.
 *
 * Evaluation (from the acting tribe's view; weights in W_*):
 * - game result: win +WIN, loss -WIN; holding every capital counts as a win (CAPITALS mode);
 * - material: our units' star cost minus visible enemy units', each scaled by remaining health;
 * - cities: our cities minus theirs;
 * - siege: our units on enemy city centres (a city is captured from there next turn), worth more on a capital;
 * - danger: next turn, which of our units the visible enemy units could kill. Each enemy unit's reachable tiles
 *   come from the engine's own pathfinder on the searched state, so zones of control and units in the way count:
 *   blocking a path (e.g. a Rider between an enemy and our besieging unit) removes the threat. If the summed damage
 *   the enemies in reach can deal (engine damage formula) reaches a unit's health, its value is counted as lost;
 *   a threatened besieger also loses its siege value. Enemy units that can reach our undefended capital cost
 *   CAPITAL_THREAT each. (Damage from several attackers is summed independently: an upper bound, a quick check.)
 *   In a fogged copy enemy techs are unknown, so enemy reach ignores e.g. Climbing: a conservative estimate.
 */
public class CombatTactics {

    public static final String[] WEIGHT_NAMES = {"win", "material", "cities", "siege", "siege_capital", "danger", "capital_threat"};
    public static final double[] DEFAULT_WEIGHTS = {10000.0, 1.0, 8.0, 3.0, 10.0, 1.0, 6.0};

    private final int me;
    private final double wWin, wMaterial, wCities, wSiege, wSiegeCapital, wDanger, wCapitalThreat;
    private final int tacticalRadius;

    public CombatTactics(int me, double[] weights, int tacticalRadius) {
        double[] w = (weights == null || weights.length != WEIGHT_NAMES.length) ? DEFAULT_WEIGHTS : weights;
        this.me = me;
        this.wWin = w[0]; this.wMaterial = w[1]; this.wCities = w[2]; this.wSiege = w[3];
        this.wSiegeCapital = w[4]; this.wDanger = w[5]; this.wCapitalThreat = w[6];
        this.tacticalRadius = tacticalRadius;
    }

    // ---- evaluation -------------------------------------------------------------------------------------------

    private static double healthValue(Unit u) {
        return u.getType().getCost() * (0.5 + 0.5 * ((double) u.getCurrentHP() / Math.max(1, u.getMaxHP())));
    }

    private List<Unit> units(GameState s, boolean ours) {
        List<Unit> out = new ArrayList<>();
        for (Actor a : s.getBoard().getActors()) {
            if (a instanceof Unit && ((a.getTribeId() == me) == ours)) out.add((Unit) a);
        }
        return out;
    }

    private List<City> cities(GameState s, boolean ours) {
        List<City> out = new ArrayList<>();
        for (Actor a : s.getBoard().getActors()) {
            if (a instanceof City && ((a.getTribeId() == me) == ours)) out.add((City) a);
        }
        return out;
    }

    /** Cheap part of the value: result, material, cities, sieges (no pathfinding). */
    public double quickValue(GameState s) {
        if (s.isGameOver()) {
            Types.RESULT r = s.getTribe(me).getWinner();
            if (r == Types.RESULT.WIN) return wWin;
            if (r == Types.RESULT.LOSS) return -wWin;
        }
        // CAPITALS mode: holding every capital wins (the engine declares it when the turn ends).
        if (holdsAllCapitals(s)) return wWin;
        double v = 0.0;
        for (Unit u : units(s, true)) v += wMaterial * healthValue(u);
        for (Unit u : units(s, false)) v -= wMaterial * healthValue(u);
        v += wCities * (cities(s, true).size() - cities(s, false).size());
        Board b = s.getBoard();
        for (City c : cities(s, false)) {
            Unit on = b.getUnitAt(c.getPosition().x, c.getPosition().y);
            if (on != null && on.getTribeId() == me) v += c.isCapital() ? wSiegeCapital : wSiege;
        }
        return v;
    }

    private boolean holdsAllCapitals(GameState s) {
        int[] caps = s.getBoard().getCapitalIDs();
        if (caps == null || caps.length < 2) return false;
        for (int id : caps) {
            Actor c = s.getActor(id);
            if (c == null || c.getTribeId() != me) return false;
        }
        return true;
    }

    /** Full value: quickValue minus next-turn danger to our units and capital. */
    public double value(GameState s) {
        double v = quickValue(s);
        if (s.isGameOver() || Math.abs(v) >= wWin) return v;
        Board b = s.getBoard();
        List<Unit> ours = units(s, true);
        List<Unit> theirs = units(s, false);
        // In a copy made with our fog the other tribes' vision is blanked, and the pathfinder never enters tiles a
        // tribe has not seen. Let them move wherever we can see (this copy is only evaluated, never played on).
        boolean[][] mine = s.getTribe(me).getObsGrid();
        for (core.actors.Tribe t : s.getTribes()) {
            if (t.getTribeId() == me) continue;
            boolean[][] g = t.getObsGrid();
            for (int x = 0; x < g.length && x < mine.length; x++) System.arraycopy(mine[x], 0, g[x], 0, Math.min(g[x].length, mine[x].length));
        }
        // Tiles each enemy unit can reach next turn (engine pathfinder on this state: our units block and exert ZOC).
        List<Set<Vector2d>> reach = new ArrayList<>();
        for (Unit e : theirs) {
            Set<Vector2d> tiles = new HashSet<>();
            tiles.add(e.getPosition().copy());
            try {
                Pathfinder pf = new Pathfinder(e.getPosition(), new core.actions.unitactions.StepMove(s, e));
                for (PathNode n : pf.findPaths()) tiles.add(n.getPosition().copy());
            } catch (RuntimeException ignored) { }
            reach.add(tiles);
        }
        for (Unit u : ours) {
            double damage = 0.0;
            for (int i = 0; i < theirs.size(); i++) {
                Unit e = theirs.get(i);
                if (e.getType() == Types.UNIT.CLOAK || e.getType() == Types.UNIT.MIND_BENDER) continue;
                boolean inReach = false;
                for (Vector2d t : reach.get(i)) {
                    if (Vector2d.chebychevDistance(t, u.getPosition()) <= e.RANGE) { inReach = true; break; }
                }
                if (!inReach) continue;
                Pair<Integer, Integer> res = AttackCommand.computeAttackResults(e, u, s, false);
                damage += res.getFirst();
            }
            if (damage >= u.getCurrentHP()) {
                double lost = wMaterial * healthValue(u);
                Vector2d p = u.getPosition();
                if (b.getTerrainAt(p.x, p.y) == Types.TERRAIN.CITY) {
                    Actor c = s.getActor(b.getCityIdAt(p.x, p.y));
                    if (c instanceof City && c.getTribeId() != me) lost += ((City) c).isCapital() ? wSiegeCapital : wSiege;
                }
                v -= wDanger * lost;
            }
        }
        for (City c : cities(s, true)) {
            if (!c.isCapital()) continue;
            Vector2d p = c.getPosition();
            if (b.getUnitAt(p.x, p.y) != null) continue;
            for (Set<Vector2d> tiles : reach) if (tiles.contains(p)) v -= wCapitalThreat;
        }
        return v;
    }

    // ---- tactical action filter ---------------------------------------------------------------------------------

    /** Attacks, city captures, infiltrations, and moves of units near the enemy (or to tiles near it). */
    public boolean isTactical(GameState s, Action a) {
        switch (a.getActionType()) {
            case ATTACK:
            case INFILTRATE:
                return true;
            case CAPTURE:
                return ((core.actions.unitactions.Capture) a).getCaptureType() == Types.TERRAIN.CITY;
            case MOVE: {
                core.actions.unitactions.Move m = (core.actions.unitactions.Move) a;
                Actor u = s.getActor(m.getUnitId());
                if (u == null) return false;
                return nearEnemy(s, u.getPosition()) || nearEnemy(s, m.getDestination());
            }
            default:
                return false;
        }
    }

    private boolean nearEnemy(GameState s, Vector2d p) {
        if (p == null) return false;
        for (Actor a : s.getBoard().getActors()) {
            if (a.getTribeId() == me || a.getTribeId() < 0) continue;
            if (Vector2d.chebychevDistance(a.getPosition(), p) <= tacticalRadius) return true;
        }
        return false;
    }

    // ---- beam search -------------------------------------------------------------------------------------------

    private static final class Node {
        final GameState state; final int rootIndex; final double quick;
        Node(GameState state, int rootIndex, double quick) { this.state = state; this.rootIndex = rootIndex; this.quick = quick; }
    }

    /**
     * @param root     copy of the live state to search on (normally with the active tribe's fog)
     * @param rootActs the live action list (indices are what the caller steps with)
     * @param allowed  raw indices the caller may choose
     * @return the raw index of the first action of the best tactical sequence, or -1 if stopping now is as good
     */
    public int chooseIndex(GameState root, ArrayList<Action> rootActs, int[] allowed, int depth, int beam, long seed) {
        Set<Integer> allowedSet = new HashSet<>();
        for (int i : allowed) allowedSet.add(i);
        double stopValue = value(root);
        List<Node> frontier = new ArrayList<>();
        // Depth 1 from the live action list, so the returned index is the caller's.
        for (int i = 0; i < rootActs.size(); i++) {
            if (!allowedSet.contains(i)) continue;
            Action a = rootActs.get(i);
            if (!isTactical(root, a)) continue;
            Node n = apply(root, a, i, seed * 1000003L + i);
            if (n != null) frontier.add(n);
        }
        if (frontier.isEmpty()) return -1;
        List<Node> leaves = new ArrayList<>(frontier);
        for (int d = 1; d < depth && !frontier.isEmpty(); d++) {
            frontier.sort(Comparator.comparingDouble((Node n) -> -n.quick));
            List<Node> kept = frontier.subList(0, Math.min(beam, frontier.size()));
            List<Node> next = new ArrayList<>();
            for (Node parent : kept) {
                if (parent.state.isGameOver() || parent.state.getActiveTribeID() != me) continue;
                ArrayList<Action> acts = parent.state.getAllAvailableActions();
                for (int j = 0; j < acts.size(); j++) {
                    Action a = acts.get(j);
                    if (!isTactical(parent.state, a)) continue;
                    Node n = apply(parent.state, a, parent.rootIndex, seed * 1000003L + (d + 1) * 7919L + j);
                    if (n != null) next.add(n);
                }
            }
            leaves.addAll(next);
            frontier = next;
        }
        // Full evaluation (with danger) only for the best few leaves by quick value.
        leaves.sort(Comparator.comparingDouble((Node n) -> -n.quick));
        int bestIndex = -1;
        double bestValue = stopValue + 1e-6;
        for (Node n : leaves.subList(0, Math.min(Math.max(beam * 2, 8), leaves.size()))) {
            double v = value(n.state);
            if (v > bestValue) { bestValue = v; bestIndex = n.rootIndex; }
        }
        return bestIndex;
    }

    private Node apply(GameState s, Action a, int rootIndex, long seed) {
        try {
            GameState c = s.copy();
            c.getRandomGenerator().setSeed(seed);
            if (!a.isFeasible(c)) return null;
            c.advance(a, true);
            return new Node(c, rootIndex, quickValue(c));
        } catch (RuntimeException e) {
            return null;
        }
    }
}
