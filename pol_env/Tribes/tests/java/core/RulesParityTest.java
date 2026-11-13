package core;

import core.actions.cityactions.BurnForest;
import core.actions.cityactions.Build;
import core.actions.cityactions.ClearForest;
import core.actions.cityactions.Destroy;
import core.actions.cityactions.GrowForest;
import core.actions.cityactions.LevelUp;
import core.actions.cityactions.command.BuildCommand;
import core.actions.cityactions.command.ClearForestCommand;
import core.actions.cityactions.command.LevelUpCommand;
import core.actions.unitactions.Examine;
import core.actions.unitactions.command.ExamineCommand;
import core.actors.City;
import core.actors.Tribe;
import core.actors.units.Knight;
import core.actors.units.Unit;
import core.game.Board;
import core.game.GameState;
import core.game.PythonEnv;
import utils.Vector2d;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

/**
 * Checks every rule switched by RulesVersion (docs/rules-parity.md) under both rule sets.
 *
 * Run from pol_env/Tribes with the compiled engine on the classpath:
 *   java -cp out:lib/json.jar:<test classes> core.RulesParityTest levels/.../map.csv levels/.../corner_map.csv
 * The second level must have its capital within two tiles of a map corner; the third must have two tribes.
 */
public final class RulesParityTest {

    private static String levelFile;
    private static String cornerLevelFile;
    private static String duelLevelFile;
    private static int checks = 0;

    private RulesParityTest() {}

    public static void main(String[] args) {
        if (args.length < 3)
            throw new IllegalArgumentException(
                    "usage: RulesParityTest <phase1 level csv> <level with capital next to a corner> <two-tribe level csv>");
        levelFile = args[0];
        cornerLevelFile = args[1];
        duelLevelFile = args[2];
        RulesVersion original = RulesVersion.active();
        try {
            for (RulesVersion v : RulesVersion.values()) {
                RulesVersion.setActiveForTesting(v);
                boolean modern = v == RulesVersion.POLYTOPIA_2_15;
                testConstants(modern);
                testLiteracy(modern);
                testClearForest(modern);
                testLumberHut(modern);
                testForestBuildingsBlockForestActions(modern);
                testPark(modern);
                testRuinUnit(modern);
                testTechRequirements(modern);
                testLighthousesStartFogged(modern);
                testLighthouseDiscovery(modern);
                testMeetingStars(modern);
                if (modern) testExplorerScores();
                testExplorerSeeksFog(modern);
                testPushDirections(modern);
                testNoSelfMeeting(modern);
            }
            testRulesIds();
        } finally {
            RulesVersion.setActiveForTesting(original);
        }
        System.out.println("RulesParityTest: PASS (" + checks + " checks)");
    }

    // ----- rules that are plain values -----

    private static void testConstants(boolean modern) {
        assertEq(modern ? 3 : 2, Types.BUILDING.LUMBER_HUT.getCost(), "Lumber Hut cost");
        assertEq(modern ? 1 : 2, TribesConfig.clearForestStars(), "Clear Forest stars");
        assertEq(modern ? 3 : 2, TribesConfig.roadCost(), "Road cost");
        assertEq(modern ? 12 : 15, TribesConfig.explorerSteps(), "Explorer steps");
        assertEq(modern ? 3 : 5, TribesConfig.burnForestCost(), "Burn Forest cost");
        assertEq(modern ? 10 : 15, new Knight(new Vector2d(0, 0), 0, false, -1, 0).getMaxHP(), "Knight max HP");
        assertEq(!modern, Types.UNIT.SWORDMAN.canFortify(), "Swordsman fortify");
        assertEq(true, Types.UNIT.DEFENDER.canFortify(), "Defender fortify (unchanged)");
        assertEq(modern, Types.BUILDING.FORGE.getTerrainRequirements().contains(Types.TERRAIN.FOREST), "Forge on forest");
        assertEq(true, Types.BUILDING.FORGE.getTerrainRequirements().contains(Types.TERRAIN.PLAIN), "Forge on plain");
        assertEq(modern ? Types.TECHNOLOGY.CLIMBING : Types.TECHNOLOGY.MEDITATION,
                TribesConfig.mountainDefenceTech(), "Mountain defence tech");
        // Unchanged values stay unchanged.
        assertEq(5, Types.BUILDING.FARM.getCost(), "Farm cost (unchanged)");
        assertEq(5, Types.BUILDING.SAWMILL.getCost(), "Sawmill cost (unchanged)");
        assertEq(2, Types.RESOURCE.ANIMAL.getCost(), "Hunt cost (unchanged)");
    }

    private static void testLiteracy(boolean modern) {
        TechnologyTree plain = new TechnologyTree();
        TechnologyTree literate = new TechnologyTree();
        literate.doResearchInit(Types.TECHNOLOGY.PHILOSOPHY);
        // Fandom "Technology Costs" table: cities -> {T1, T2, T3} with Literacy.
        int[][] table = {
                {1, 4, 4, 5}, {2, 4, 6, 7}, {3, 5, 7, 9}, {4, 6, 8, 11},
                {5, 6, 10, 13}, {6, 7, 11, 15}, {7, 8, 12, 17}, {8, 8, 14, 19},
        };
        Types.TECHNOLOGY[] byTier = {Types.TECHNOLOGY.CLIMBING, Types.TECHNOLOGY.MINING, Types.TECHNOLOGY.SMITHERY};
        for (int[] row : table) {
            int cities = row[0];
            for (int tier = 1; tier <= 3; tier++) {
                int base = TribesConfig.TECH_BASE_COST + tier * cities;
                assertEq(base, byTier[tier - 1].getCost(cities, plain), "tech cost without Literacy");
                int expected = modern ? row[tier] : (int) (base * TribesConfig.TECH_DISCOUNT_VALUE);
                assertEq(expected, byTier[tier - 1].getCost(cities, literate),
                        "tech cost with Literacy, tier " + tier + ", " + cities + " cities");
            }
        }
    }

    // ----- rules that need a game -----

    private static GameState freshGame() {
        PythonEnv env = new PythonEnv();
        env.initFromLevel(levelFile, 1234L, Types.GAME_MODE.SCORE);
        return env.getGameState();
    }

    private static City capital(GameState gs) {
        Tribe t = gs.getTribe(0);
        return (City) gs.getActor(t.getCapitalID());
    }

    private static Vector2d findCityTile(GameState gs, City city, Types.TERRAIN terrain, boolean mustBeEmpty) {
        Board b = gs.getBoard();
        for (Vector2d p : b.getCityTiles(city.getActorId())) {
            if (b.getTerrainAt(p.x, p.y) != terrain) continue;
            if (mustBeEmpty && (b.getBuildingAt(p.x, p.y) != null || b.getUnitAt(p.x, p.y) != null)) continue;
            return p;
        }
        throw new AssertionError("level has no " + terrain + " tile in the capital's territory");
    }

    private static void testClearForest(boolean modern) {
        GameState gs = freshGame();
        Tribe t = gs.getTribe(0);
        City c = capital(gs);
        t.getTechTree().doResearchInit(Types.TECHNOLOGY.FORESTRY);
        Vector2d forest = findCityTile(gs, c, Types.TERRAIN.FOREST, true);
        ClearForest a = new ClearForest(c.getActorId());
        a.setTargetPos(forest);
        int before = t.getStars();
        assertEq(true, new ClearForestCommand().execute(a, gs), "Clear Forest executes");
        assertEq(before + (modern ? 1 : 2), t.getStars(), "stars after Clear Forest");
        assertEq(Types.TERRAIN.PLAIN, gs.getBoard().getTerrainAt(forest.x, forest.y), "cleared tile is plain");
    }

    private static void testLumberHut(boolean modern) {
        GameState gs = freshGame();
        Tribe t = gs.getTribe(0);
        City c = capital(gs);
        t.getTechTree().doResearchInit(Types.TECHNOLOGY.FORESTRY);
        t.setStars(10);
        Vector2d forest = findCityTile(gs, c, Types.TERRAIN.FOREST, true);
        Build a = new Build(c.getActorId());
        a.setBuildingType(Types.BUILDING.LUMBER_HUT);
        a.setTargetPos(forest);
        int pop = c.getPopulation();
        assertEq(true, new BuildCommand().execute(a, gs), "Lumber Hut builds");
        assertEq(10 - (modern ? 3 : 2), t.getStars(), "stars after Lumber Hut");
        assertEq(pop + 1, c.getPopulation(), "Lumber Hut population (unchanged)");

        // With 2 stars a Lumber Hut is affordable only under the 2020 rules.
        t.setStars(2);
        Vector2d other = findCityTile(gs, c, Types.TERRAIN.FOREST, true);
        Build b = new Build(c.getActorId());
        b.setBuildingType(Types.BUILDING.LUMBER_HUT);
        b.setTargetPos(other);
        assertEq(!modern, b.isFeasible(gs), "Lumber Hut feasible with 2 stars");
    }

    private static void testForestBuildingsBlockForestActions(boolean modern) {
        // A forest tile carrying a building (here a Forest Temple) can be cleared or burnt only under 2020 rules.
        GameState gs = freshGame();
        Tribe t = gs.getTribe(0);
        City c = capital(gs);
        t.getTechTree().doResearchInit(Types.TECHNOLOGY.FORESTRY);
        t.getTechTree().doResearchInit(Types.TECHNOLOGY.CHIVALRY);
        t.getTechTree().doResearchInit(Types.TECHNOLOGY.CONSTRUCTION);
        t.setStars(20);
        Vector2d forest = findCityTile(gs, c, Types.TERRAIN.FOREST, true);
        gs.getBoard().setBuildingAt(forest.x, forest.y, Types.BUILDING.FOREST_TEMPLE);
        ClearForest clear = new ClearForest(c.getActorId());
        clear.setTargetPos(forest);
        assertEq(!modern, clear.isFeasible(gs), "Clear Forest on a forest with a building");
        BurnForest burn = new BurnForest(c.getActorId());
        burn.setTargetPos(forest);
        assertEq(!modern, burn.isFeasible(gs), "Burn Forest on a forest with a building");

        Vector2d plain = findCityTile(gs, c, Types.TERRAIN.PLAIN, true);
        t.getTechTree().doResearchInit(Types.TECHNOLOGY.SPIRITUALISM);
        GrowForest grow = new GrowForest(c.getActorId());
        grow.setTargetPos(plain);
        assertEq(true, grow.isFeasible(gs), "Grow Forest on an empty plain");
        gs.getBoard().setBuildingAt(plain.x, plain.y, Types.BUILDING.SAWMILL);
        assertEq(!modern, grow.isFeasible(gs), "Grow Forest on a plain with a building");
    }

    private static void testPark(boolean modern) {
        GameState gs = freshGame();
        City c = capital(gs);
        while (c.getLevel() < 4) c.levelUp();
        c.setPopulation(c.getPopulation_need());
        int prod = c.getProduction();
        LevelUp a = new LevelUp(c.getActorId());
        a.setBonus(Types.CITY_LEVEL_UP.PARK);
        assertEq(true, new LevelUpCommand().execute(a, gs), "Park level-up executes");
        // +1 for the level itself; +1 more from the Park under 2.15 rules.
        assertEq(prod + 1 + (modern ? 1 : 0), c.getProduction(), "production after Park");
    }

    private static void testRuinUnit(boolean modern) {
        // The ruin reward is random; try seeds until the unit reward comes up.
        for (long seed = 0; seed < 200; seed++) {
            PythonEnv env = new PythonEnv();
            env.initFromLevel(levelFile, seed, Types.GAME_MODE.SCORE);
            GameState gs = env.getGameState();
            Tribe t = gs.getTribe(0);
            ArrayList<Unit> units = gs.getUnits(0);
            if (units.isEmpty()) throw new AssertionError("level has no starting unit");
            Unit explorer = units.get(0);
            Vector2d pos = explorer.getPosition().copy();
            if (gs.getBoard().getTerrainAt(pos.x, pos.y).isWater()) continue;
            gs.getBoard().setResourceAt(pos.x, pos.y, Types.RESOURCE.RUINS);
            explorer.setStatus(Types.TURN_STATUS.FRESH);
            Set<Integer> before = new HashSet<>();
            for (Unit u : gs.getUnits(0)) before.add(u.getActorId());
            int stars = t.getStars();
            int techs = t.getTechTree().getNumResearched();
            Examine ex = new Examine(explorer.getActorId());
            assertEq(true, new ExamineCommand().execute(ex, gs), "Examine executes");
            Unit added = null;
            for (Unit u : gs.getUnits(0)) if (!before.contains(u.getActorId()) && u.getPosition().equals(pos)) added = u;
            if (added == null) continue;
            // The unit reward gives nothing else (in particular no meeting stars from the pushed unit's view).
            assertEq(stars, t.getStars(), "stars after the ruin unit reward");
            assertEq(techs, t.getTechTree().getNumResearched(), "techs after the ruin unit reward");
            if (modern) {
                assertEq(Types.UNIT.SWORDMAN, added.getType(), "ruin unit on land");
                assertEq(true, added.isVeteran(), "ruin Swordsman is veteran");
                int hp = TribesConfig.SWORDMAN_MAX_HP + TribesConfig.VETERAN_PLUS_HP;
                assertEq(hp, added.getMaxHP(), "ruin Swordsman max HP");
                assertEq(hp, added.getCurrentHP(), "ruin Swordsman current HP");
            } else {
                assertEq(Types.UNIT.SUPERUNIT, added.getType(), "ruin unit on land");
                assertEq(false, added.isVeteran(), "ruin Giant is not veteran");
            }
            return;
        }
        throw new AssertionError("no seed produced the ruin unit reward");
    }

    private static void testTechRequirements(boolean modern) {
        GameState gs = freshGame();
        Tribe t = gs.getTribe(0);
        City c = capital(gs);
        t.setStars(20);
        Vector2d forest = findCityTile(gs, c, Types.TERRAIN.FOREST, true);
        BurnForest burn = new BurnForest(c.getActorId());
        burn.setTargetPos(forest);
        t.getTechTree().doResearchInit(Types.TECHNOLOGY.CONSTRUCTION);
        assertEq(modern, burn.isFeasible(gs), "Burn Forest with Construction only");
        assertEq(modern ? Types.TECHNOLOGY.CONSTRUCTION : Types.TECHNOLOGY.CHIVALRY,
                Types.ACTION.BURN_FOREST.getTechnologyRequirement(), "BURN_FOREST action tech");
        assertEq(modern ? Types.TECHNOLOGY.CHIVALRY : Types.TECHNOLOGY.CONSTRUCTION,
                Types.ACTION.DESTROY.getTechnologyRequirement(), "DESTROY action tech");

        Vector2d plain = findCityTile(gs, c, Types.TERRAIN.PLAIN, true);
        t.getTechTree().doResearchInit(Types.TECHNOLOGY.FORESTRY);
        t.getTechTree().doResearchInit(Types.TECHNOLOGY.MATHEMATICS);
        Build saw = new Build(c.getActorId());
        saw.setBuildingType(Types.BUILDING.SAWMILL);
        saw.setTargetPos(plain);
        gs.getBoard().setBuildingAt(plain.x, plain.y, Types.BUILDING.SAWMILL);
        c.addBuilding(gs, new core.actors.Building(plain.x, plain.y, Types.BUILDING.SAWMILL, c.getActorId()));
        Destroy destroy = new Destroy(c.getActorId());
        destroy.setTargetPos(plain);
        assertEq(!modern, destroy.isFeasible(gs), "Destroy with Construction only");
    }

    private static boolean[] cornersVisible(Tribe t) {
        boolean[][] g = t.getObsGrid();
        int n = g.length;
        return new boolean[]{g[0][0], g[0][n - 1], g[n - 1][0], g[n - 1][n - 1]};
    }

    private static void testLighthousesStartFogged(boolean modern) {
        PythonEnv env = new PythonEnv();
        env.initFromLevel(cornerLevelFile, 1234L, Types.GAME_MODE.SCORE);
        GameState gs = env.getGameState();
        Tribe t = gs.getTribe(0);
        boolean anyVisible = false;
        for (boolean v : cornersVisible(t)) anyVisible |= v;
        // The capital's starting view covers a corner: visible under 2020 rules, fogged under 2.15 rules.
        assertEq(!modern, anyVisible, "a corner is visible at the start");
        if (modern) assertEq(0, t.countLighthousesFound(), "lighthouses found at the start");
        assertEq(0, capital(gs).getPopulation(), "capital population at the start");
    }

    private static void testLighthouseDiscovery(boolean modern) {
        GameState gs = freshGame();
        Tribe t = gs.getTribe(0);
        Board b = gs.getBoard();
        City c = capital(gs);
        int n = b.getSize();
        int[][] corners = {{0, 0}, {0, n - 1}, {n - 1, 0}, {n - 1, n - 1}};
        int pop = c.getPopulation();
        assertEq(Types.BUILDING.MONUMENT_STATUS.UNAVAILABLE, t.getMonuments().get(Types.BUILDING.EYE_OF_GOD),
                "Eye of God before the lighthouses");
        int found = 0;
        for (int[] corner : corners) {
            boolean wasVisible = t.getObsGrid()[corner[0]][corner[1]];
            t.clearView(corner[0], corner[1], 1, gs.getRandomGenerator(), b);
            t.applyDeferredViewEffects(b);
            if (!wasVisible) found++;
            assertEq(pop + (modern ? found : 0), c.getPopulation(), "capital population after lighthouse " + found);
            // Seeing the same corner again gives nothing more.
            t.clearView(corner[0], corner[1], 1, gs.getRandomGenerator(), b);
            t.applyDeferredViewEffects(b);
            assertEq(pop + (modern ? found : 0), c.getPopulation(), "capital population after re-seeing a lighthouse");
        }
        assertEq(4, found, "all corners were fogged");
        assertEq(modern ? Types.BUILDING.MONUMENT_STATUS.AVAILABLE : Types.BUILDING.MONUMENT_STATUS.UNAVAILABLE,
                t.getMonuments().get(Types.BUILDING.EYE_OF_GOD), "Eye of God after all four lighthouses");
        assertEq(modern ? null : Types.TECHNOLOGY.NAVIGATION, Types.BUILDING.EYE_OF_GOD.getTechnologyRequirement(),
                "Eye of God tech");
    }

    private static void testMeetingStars(boolean modern) {
        PythonEnv env = new PythonEnv();
        env.initFromLevel(duelLevelFile, 1234L, Types.GAME_MODE.CAPITALS);
        GameState gs = env.getGameState();
        Board b = gs.getBoard();
        Tribe me = gs.getTribe(0);
        Tribe other = gs.getTribe(1);
        assertEq(0, b.getActiveTribeID(), "tribe 0 is to move");
        assertEq(false, me.getTribesMet().contains(1), "tribes have not met at the start");
        assertEq(false, other.getTribesMet().contains(0), "tribes have not met at the start (other side)");
        City otherCapital = (City) gs.getActor(other.getCapitalID());
        Vector2d cap = otherCapital.getPosition();

        // Off-turn: under 2.15 rules a view cleared during the other tribe's turn does not meet anyone.
        b.setActiveTribeID(1);
        int myStars = me.getStars();
        me.clearView(cap.x, cap.y, 1, gs.getRandomGenerator(), b);
        assertEq(!modern, me.getTribesMet().contains(1), "tribe 0 meets tribe 1 off-turn");
        b.setActiveTribeID(0);
        if (modern) {
            // On tribe 0's turn, seeing tribe 1's territory meets it.
            int otherScore = other.getScore();
            me.clearView(cap.x, cap.y, 1, gs.getRandomGenerator(), b);
            assertEq(true, me.getTribesMet().contains(1), "tribe 0 meets tribe 1 on its turn");
            assertEq(myStars + meetingStars(otherScore), me.getStars(), "stars for tribe 0 (moves first: no bonus)");
        }

        // Being seen is not meeting: tribe 1 has not met tribe 0 yet.
        assertEq(false, other.getTribesMet().contains(0), "being seen does not meet");

        // Put a tribe-0 unit next to tribe 1's capital, then start tribe 1's turn: under 2.15 rules it meets.
        Unit mine = gs.getUnits(0).get(0);
        Vector2d spot = null;
        for (Vector2d n : cap.neighborhood(1, 0, b.getSize()))
            if (b.getUnitAt(n.x, n.y) == null && b.getTerrainAt(n.x, n.y) != Types.TERRAIN.MOUNTAIN
                    && !b.getTerrainAt(n.x, n.y).isWater()) { spot = n; break; }
        if (spot == null) throw new AssertionError("no free tile next to tribe 1's capital");
        b.moveUnit(mine, mine.getPosition().x, mine.getPosition().y, spot.x, spot.y, gs.getRandomGenerator());
        int otherStars = other.getStars();
        int myScore = me.getScore();
        other.meetVisibleTribes(b);
        assertEq(modern, other.getTribesMet().contains(0), "tribe 1 meets tribe 0 at the start of its turn");
        // Tribe 1 moves second, so under 2.15 rules it also gets the +2 second-player bonus.
        assertEq(otherStars + (modern ? meetingStars(myScore) + 2 : 0), other.getStars(), "stars for tribe 1");
    }

    /** A game where everything is visible except the given tiles, with Climbing so every land tile is walkable. */
    private static GameState gameWithFog(int[][] fogged) {
        GameState gs = freshGame();
        Tribe t = gs.getTribe(0);
        t.getTechTree().doResearchInit(Types.TECHNOLOGY.CLIMBING);
        boolean[][] g = t.getObsGrid();
        for (boolean[] row : g) java.util.Arrays.fill(row, true);
        for (int[] f : fogged) g[f[0]][f[1]] = false;
        return gs;
    }

    private static void testExplorerScores() {
        // Phase 1 maps are all land, so every tile is walkable with Climbing.
        GameState gs = gameWithFog(new int[][]{{0, 0}, {6, 8}});
        Board b = gs.getBoard();
        int[][] sc = b.explorerScores(0);
        assertEq(140, sc[1][1], "score: 1 fog tile that is a lighthouse");
        assertEq(173, sc[5][7], "score: 1 plain fog tile");
        assertEq(240, sc[2][2], "score: one step from a lighthouse tile");
        assertEq(273, sc[4][6], "score: one step from a plain fog tile");
        assertEq(440, sc[4][4], "score: three steps from a lighthouse tile (BFS limit)");
        assertEq(Integer.MAX_VALUE, sc[10][0], "score: out of BFS reach");

        // Four or more fog tiles share the best score; a lighthouse beats any plain count.
        gs = gameWithFog(new int[][]{{4, 4}, {4, 5}, {4, 6}, {5, 4}, {5, 6}, {10, 10}});
        sc = gs.getBoard().explorerScores(0);
        assertEq(143, sc[5][5], "score: 5 fog tiles counts as 4+");
        assertEq(153, sc[3][5], "score: 3 fog tiles");
        assertEq(140, sc[9][9], "score: lighthouse with 1 tile beats 4+ plain tiles");
    }

    private static void testExplorerSeeksFog(boolean modern) {
        // One fogged tile three steps from the capital: the 2.15 explorer walks straight to it.
        GameState gs = freshGame();
        City c = capital(gs);
        Vector2d p = c.getPosition();
        int n = gs.getBoard().getSize();
        int fx = p.x + 3 < n ? p.x + 3 : p.x - 3;
        GameState g2 = gameWithFog(new int[][]{{fx, p.y}});
        Tribe t = g2.getTribe(0);
        g2.getBoard().launchExplorer(p.x, p.y, 0, new java.util.Random(5));
        if (modern) assertEq(true, t.getObsGrid()[fx][p.y], "2.15 explorer reveals the only fog tile");
        // Under 2020 rules the random walk may or may not find it; only the 2.15 behaviour is checked.
        checks++;
    }

    /** Places a new extra (cityless) unit of tribe 0 at (x, y) with the given last direction. */
    private static Unit placeUnit(GameState gs, Types.UNIT type, int x, int y, int dx, int dy) {
        Unit u = Types.UNIT.createUnit(new Vector2d(x, y), 0, false, -1, 0, type);
        gs.getBoard().addUnit(capital(gs), u);
        u.setLastDirection(dx, dy);
        return u;
    }

    private static GameState emptyBoardGame() {
        // Climbing makes every Phase 1 tile walkable; remove the starting units so tiles are free.
        GameState gs = freshGame();
        gs.getTribe(0).getTechTree().doResearchInit(Types.TECHNOLOGY.CLIMBING);
        for (Unit u : gs.getUnits(0)) gs.getBoard().removeUnitFromBoard(u);
        return gs;
    }

    private static void assertPushedTo(GameState gs, Unit u, int spawner, int ex, int ey, String what) {
        Vector2d p = u.getPosition().copy();
        gs.pushUnit(u, p.x, p.y, spawner);
        assertEq(new Vector2d(ex, ey), u.getPosition(), what);
    }

    private static void testPushDirections(boolean modern) {
        // Map is 11x11, centre (5, 5); x is the row, south is +x.
        GameState gs = emptyBoardGame();
        Unit u = placeUnit(gs, Types.UNIT.WARRIOR, 2, 3, 0, 0);
        if (!modern) {
            // 2020 rules: fixed order, first (0, +1).
            assertPushedTo(gs, u, 0, 2, 4, "2020 push goes to (0,+1) first");
            assertEq(0, u.getLastDirX() * 10 + u.getLastDirY(), "2020 rules never set a direction");
            return;
        }
        assertPushedTo(gs, u, 0, 3, 4, "unmoved unit is pushed towards the centre");
        assertEq(11, u.getLastDirX() * 10 + u.getLastDirY(), "a push sets the direction");

        gs = emptyBoardGame();
        u = placeUnit(gs, Types.UNIT.WARRIOR, 5, 5, 0, 0);
        assertPushedTo(gs, u, 0, 6, 5, "unmoved unit on the exact centre is pushed south");

        gs = emptyBoardGame();
        u = placeUnit(gs, Types.UNIT.WARRIOR, 4, 4, -1, 0);
        assertPushedTo(gs, u, 0, 3, 4, "friendly unit keeps its direction");

        gs = emptyBoardGame();
        u = placeUnit(gs, Types.UNIT.WARRIOR, 4, 4, -1, 0);
        assertPushedTo(gs, u, 1, 5, 4, "enemy unit is pushed back the opposite way");

        gs = emptyBoardGame();
        u = placeUnit(gs, Types.UNIT.ARCHER, 4, 4, -1, 0);
        assertPushedTo(gs, u, 1, 3, 4, "ranged unit keeps its direction even when pushed by an enemy");

        gs = emptyBoardGame();
        u = placeUnit(gs, Types.UNIT.WARRIOR, 4, 4, -1, 0);
        placeUnit(gs, Types.UNIT.WARRIOR, 3, 4, 0, 0);
        assertPushedTo(gs, u, 0, 3, 3, "blocked: tries counterclockwise first (north -> north-west)");

        gs = emptyBoardGame();
        u = placeUnit(gs, Types.UNIT.WARRIOR, 4, 4, -1, 0);
        placeUnit(gs, Types.UNIT.WARRIOR, 3, 4, 0, 0);
        placeUnit(gs, Types.UNIT.WARRIOR, 3, 3, 0, 0);
        assertPushedTo(gs, u, 0, 3, 5, "then clockwise (north -> north-east)");

        // A move sets the direction; a ranged shot is covered in AttackCommand.
        gs = emptyBoardGame();
        u = placeUnit(gs, Types.UNIT.WARRIOR, 4, 4, 0, 0);
        gs.getBoard().moveUnit(u, 4, 4, 4, 6, gs.getRandomGenerator());
        assertEq(1, u.getLastDirX() * 10 + u.getLastDirY(), "a move east sets direction (0, +1)");
        assertEq(1, u.copy(false).getLastDirX() * 10 + u.copy(false).getLastDirY(), "copies keep the direction");
    }

    private static int meetingStars(int score) {
        if (score >= 4000) return 11;
        if (score >= 3000) return 9;
        if (score >= 2000) return 7;
        if (score >= 1000) return 5;
        return 3;
    }

    private static void testNoSelfMeeting(boolean modern) {
        // Solo Phase 1 game: moving a unit must never give meeting stars.
        GameState gs = freshGame();
        Tribe t = gs.getTribe(0);
        t.getTechTree().doResearchInit(Types.TECHNOLOGY.CLIMBING);
        Unit u = gs.getUnits(0).get(0);
        Vector2d p = u.getPosition();
        int stars = t.getStars();
        for (Vector2d n : p.neighborhood(1, 0, gs.getBoard().getSize())) {
            if (gs.getBoard().getUnitAt(n.x, n.y) != null) continue;
            gs.getBoard().moveUnit(u, p.x, p.y, n.x, n.y, gs.getRandomGenerator());
            break;
        }
        assertEq(stars, t.getStars(), "stars after a move in a solo game");
        if (modern) assertEq(false, t.getTribesMet().contains(0), "a tribe does not meet itself");
    }

    private static void testRulesIds() {
        assertEq(RulesVersion.TRIBES_2020, RulesVersion.fromId(null), "default rules");
        assertEq(RulesVersion.TRIBES_2020, RulesVersion.fromId(""), "empty rules");
        assertEq(RulesVersion.POLYTOPIA_2_15, RulesVersion.fromId(" Polytopia_2_15 "), "rules id is normalised");
        try {
            RulesVersion.fromId("polytopia_3");
            throw new AssertionError("unknown rules id accepted");
        } catch (IllegalArgumentException expected) {
            checks++;
        }
    }

    private static void assertEq(Object expected, Object actual, String what) {
        checks++;
        if (expected == null ? actual != null : !expected.equals(actual)) {
            throw new AssertionError(RulesVersion.activeId() + ": " + what + ": expected " + expected + " but was " + actual);
        }
    }
}
