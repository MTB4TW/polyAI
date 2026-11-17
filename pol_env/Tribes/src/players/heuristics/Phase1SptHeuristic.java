package players.heuristics;

import core.TechnologyTree;
import core.Types;
import core.actors.City;
import core.actors.units.Unit;
import core.game.Board;
import core.game.GameState;
import utils.Vector2d;

import java.util.ArrayList;

/**
 * State evaluation aligned with the Phase 1 objective: Bardur's total city production (SPT) at the end of Turn 10.
 *
 * The value is a weighted sum of {@link #FEATURE_NAMES}. Weights are supplied by the caller (Python) so they can be
 * tuned without recompiling. When the state was produced by {@code GameState.copy(playerId)}, fogged tiles read as
 * FOG, so every feature uses only information the player can see.
 */
public class Phase1SptHeuristic implements StateHeuristic {

    public static final String[] FEATURE_NAMES = {
            "spt",              // sum of production over owned cities (the evaluation target)
            "pop_progress",     // sum over cities of population / population needed for the next level
            "stars_usable",     // banked stars, zero once no turn remains to spend them
            "units_on_village", // own units standing on an uncaptured village (capturable)
            "village_proximity",// sum over known villages of (cap - distance to the nearest own unit)
            "techs",            // technologies researched
            "cities",           // number of owned cities
            "visible_tiles",    // fraction of the board visible to the player
            "units",            // own units, zero once fewer than two turns remain to use them
            "unlocked_pop",     // population still obtainable inside own borders with the techs already researched
            "frontier_proximity", // sum over own units of (cap - distance to the nearest tile the player cannot see)
    };

    public static final int NUM_FEATURES = FEATURE_NAMES.length;
    private static final int DIST_CAP = 8;

    private final int playerID;
    private final double[] weights;
    private final int turnsRemainingAtRoot;
    private final int rootTick;

    public Phase1SptHeuristic(int playerID, double[] weights, int turnsRemainingAtRoot, int rootTick) {
        if (weights == null || weights.length != NUM_FEATURES)
            throw new IllegalArgumentException("Phase1SptHeuristic expects " + NUM_FEATURES + " weights");
        this.playerID = playerID;
        this.weights = weights.clone();
        this.turnsRemainingAtRoot = turnsRemainingAtRoot;
        this.rootTick = rootTick;
    }

    @Override
    public double evaluateState(GameState gs) {
        double[] f = features(gs);
        double v = 0.0;
        for (int i = 0; i < NUM_FEATURES; i++) v += weights[i] * f[i];
        return v;
    }

    public double[] features(GameState gs) {
        double[] f = new double[NUM_FEATURES];
        ArrayList<City> cities = gs.getCities(playerID);
        double spt = 0.0, pop = 0.0;
        for (City c : cities) {
            spt += c.getProduction();
            int need = Math.max(1, c.getPopulation_need());
            pop += Math.max(0, c.getPopulation()) / (double) need;
        }
        int turnsLeft = turnsRemainingAtRoot - (gs.getTick() - rootTick);

        Board board = gs.getBoard();
        int size = board.getSize();
        ArrayList<Vector2d> villages = new ArrayList<>();
        for (int x = 0; x < size; x++)
            for (int y = 0; y < size; y++)
                if (board.getTerrainAt(x, y) == Types.TERRAIN.VILLAGE) villages.add(new Vector2d(x, y));

        ArrayList<Unit> units = gs.getUnits(playerID);
        int onVillage = 0;
        double dist = 0.0;
        for (Vector2d v : villages) {
            int best = DIST_CAP;
            for (Unit u : units) {
                Vector2d p = u.getPosition();
                best = Math.min(best, Math.max(Math.abs(v.x - p.x), Math.abs(v.y - p.y)));
            }
            if (best == 0) onVillage++;
            dist += DIST_CAP - best;
        }

        TechnologyTree tt = gs.getTribeTechTree(playerID);
        java.util.HashSet<Integer> myCityIds = new java.util.HashSet<>();
        for (City c : cities) myCityIds.add(c.getActorId());
        double unlocked = 0.0;
        for (int x = 0; x < size; x++) {
            for (int y = 0; y < size; y++) {
                if (!myCityIds.contains(board.getCityIdAt(x, y)) || board.getBuildingAt(x, y) != null) continue;
                Types.RESOURCE r = board.getResourceAt(x, y);
                Types.TERRAIN t = board.getTerrainAt(x, y);
                if (r == Types.RESOURCE.FRUIT && tt.isResearched(Types.TECHNOLOGY.ORGANIZATION)) unlocked += 1;
                if (r == Types.RESOURCE.ANIMAL && tt.isResearched(Types.TECHNOLOGY.HUNTING)) unlocked += 1;
                if (r == Types.RESOURCE.FISH && tt.isResearched(Types.TECHNOLOGY.FISHING)) unlocked += 1;
                if (r == Types.RESOURCE.CROPS && tt.isResearched(Types.TECHNOLOGY.FARMING)) unlocked += 2;
                if (r == Types.RESOURCE.ORE && t == Types.TERRAIN.MOUNTAIN && tt.isResearched(Types.TECHNOLOGY.MINING)) unlocked += 2;
                if (t == Types.TERRAIN.FOREST && tt.isResearched(Types.TECHNOLOGY.FORESTRY)) unlocked += 1;
            }
        }

        boolean[][] vis = gs.getTribe(playerID).getObsGrid();
        int visCount = 0;
        for (boolean[] row : vis) for (boolean b : row) if (b) visCount++;

        f[0] = spt;
        f[1] = pop;
        f[2] = turnsLeft > 0 ? gs.getTribe(playerID).getStars() : 0.0;
        f[3] = onVillage;
        f[4] = dist;
        f[5] = tt.getNumResearched();
        f[6] = cities.size();
        f[7] = visCount / (double) (vis.length * vis[0].length);
        f[8] = turnsLeft >= 2 ? units.size() : 0.0;
        f[9] = unlocked;
        double frontier = 0.0;
        for (Unit u : units) {
            Vector2d p = u.getPosition();
            int best = DIST_CAP;
            for (int x = 0; x < vis.length && best > 1; x++)
                for (int y = 0; y < vis[x].length; y++)
                    if (!vis[x][y]) best = Math.min(best, Math.max(Math.abs(x - p.x), Math.abs(y - p.y)));
            frontier += DIST_CAP - best;
        }
        f[10] = frontier;
        return f;
    }
}
