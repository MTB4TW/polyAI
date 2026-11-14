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
                testMarket(modern);
                testRuinExplorerCondition(modern);
                testBackwardsResearch(modern);
                testStartingStarsAndTribes(modern);
                testDiplomacyAbsentUnder2020(modern);
                if (modern) {
                    testPeaceAndEmbassies();
                    testOfferLapses();
                    testInfiltration();
                    testHiddenCloak();
                    testDaggerSkills();
                }
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
        assertEq(modern ? Types.TERRAIN.FOREST : Types.TERRAIN.PLAIN, gs.getBoard().getTerrainAt(forest.x, forest.y),
                "Lumber Hut tile terrain");

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

    // ----- Step 2, 2.14 tribes, backwards research -----

    /** Builds `type` at `pos` for the capital through the real Build action; returns whether it was feasible. */
    private static boolean build(GameState gs, City c, Types.BUILDING type, Vector2d pos) {
        Build a = new Build(c.getActorId());
        a.setBuildingType(type);
        a.setTargetPos(pos);
        return new BuildCommand().execute(a, gs);
    }

    private static void testMarket(boolean modern) {
        GameState gs = emptyBoardGame();
        Board b = gs.getBoard();
        Tribe t = gs.getTribe(0);
        City c = capital(gs);
        for (Types.TECHNOLOGY tech : new Types.TECHNOLOGY[]{Types.TECHNOLOGY.HUNTING, Types.TECHNOLOGY.FORESTRY,
                Types.TECHNOLOGY.MATHEMATICS, Types.TECHNOLOGY.RIDING, Types.TECHNOLOGY.ROADS, Types.TECHNOLOGY.TRADE})
            t.getTechTree().doResearchInit(tech);
        t.setStars(200);
        // Pick a sawmill tile with three other city tiles next to it: two huts and a market.
        Vector2d centre = c.getPosition();
        Vector2d saw = null, hut1 = null, hut2 = null, market = null;
        for (Vector2d s0 : b.getCityTiles(c.getActorId())) {
            if (s0.equals(centre)) continue;
            ArrayList<Vector2d> adj = new ArrayList<>();
            for (Vector2d n : b.getCityTiles(c.getActorId()))
                if (!n.equals(centre) && !n.equals(s0) && Vector2d.chebychevDistance(n, s0) == 1) adj.add(n);
            if (adj.size() >= 3) { saw = s0; hut1 = adj.get(0); hut2 = adj.get(1); market = adj.get(2); break; }
        }
        if (saw == null) throw new AssertionError("no room for a market layout around the capital");
        for (Vector2d p : new Vector2d[]{saw, market}) { b.setTerrainAt(p.x, p.y, Types.TERRAIN.PLAIN); b.setResourceAt(p.x, p.y, null); }
        for (Vector2d p : new Vector2d[]{hut1, hut2}) { b.setTerrainAt(p.x, p.y, Types.TERRAIN.FOREST); b.setResourceAt(p.x, p.y, null); }

        assertEq(false, build(gs, c, Types.BUILDING.CUSTOMS_HOUSE, market), "market/customs house with nothing next to it");
        assertEq(true, build(gs, c, Types.BUILDING.LUMBER_HUT, hut1), "hut 1");
        assertEq(true, build(gs, c, Types.BUILDING.LUMBER_HUT, hut2), "hut 2");
        assertEq(true, build(gs, c, Types.BUILDING.SAWMILL, saw), "sawmill");
        int prod = c.getProduction();
        // 2020: the Customs House still needs a Port. 2.15: the Market needs a Sawmill/Windmill/Forge next to it.
        assertEq(modern, build(gs, c, Types.BUILDING.CUSTOMS_HOUSE, market), "market next to a sawmill");
        if (modern) {
            int huts = 0;
            for (Vector2d n : saw.neighborhood(1, 0, b.getSize()))
                if (b.getBuildingAt(n.x, n.y) == Types.BUILDING.LUMBER_HUT) huts++;
            assertEq(Math.min(huts, TribesConfig.MARKET_MAX_INCOME_2_15), c.getMarketIncome(), "market income = sawmill level");
            assertEq(prod + c.getMarketIncome(), c.getProduction(), "market income counts in production");
        } else {
            assertEq(0, c.getMarketIncome(), "no market income under 2020 rules");
        }
    }

    private static void testRuinExplorerCondition(boolean modern) {
        GameState gs = gameWithFog(new int[][]{{0, 0}});
        Tribe t = gs.getTribe(0);
        assertEq(true, core.actions.unitactions.command.ExamineCommand.hasUnexploredTileNearby(gs, t, new Vector2d(2, 2)),
                "fog two tiles away counts");
        assertEq(false, core.actions.unitactions.command.ExamineCommand.hasUnexploredTileNearby(gs, t, new Vector2d(3, 3)),
                "fog three tiles away does not");
    }

    private static void testBackwardsResearch(boolean modern) {
        TechnologyTree tt = new TechnologyTree();
        tt.doResearchInit(Types.TECHNOLOGY.SMITHERY);
        assertEq(modern, tt.isResearchable(Types.TECHNOLOGY.MINING), "Mining after Smithery (backwards)");
        assertEq(false, tt.isResearchable(Types.TECHNOLOGY.MEDITATION), "a sibling of Mining stays locked");
        assertEq(TribesConfig.TECH_BASE_COST + 2, Types.TECHNOLOGY.MINING.getCost(1, tt), "backwards research costs the tech's tier");
        TechnologyTree d = new TechnologyTree();
        d.doResearchInit(Types.TECHNOLOGY.ORGANIZATION);
        d.doResearchInit(Types.TECHNOLOGY.SHIELDS);
        assertEq(modern, d.isResearchable(Types.TECHNOLOGY.DIPLOMACY), "Diplomacy researchable after Strategy");
        // A tree with every 2020 tech is complete under the 2020 rules only (2.15 adds Diplomacy).
        TechnologyTree all = new TechnologyTree();
        for (Types.TECHNOLOGY tech : Types.TECHNOLOGY.values()) if (tech != Types.TECHNOLOGY.DIPLOMACY) all.doResearchInit(tech);
        assertEq(!modern, all.copy().isEverythingResearched(), "tree complete without Diplomacy");
    }

    private static void testStartingStarsAndTribes(boolean modern) {
        int[] expected2015 = new int[Types.TRIBE.values().length];
        java.util.Arrays.fill(expected2015, 5);
        expected2015[Types.TRIBE.XIN_XI.ordinal()] = 7;
        expected2015[Types.TRIBE.OUMAJI.ordinal()] = 6;
        expected2015[Types.TRIBE.HOODRICK.ordinal()] = 7;
        expected2015[Types.TRIBE.QUETZALI.ordinal()] = 7;
        expected2015[Types.TRIBE.YADAKK.ordinal()] = 7;
        expected2015[Types.TRIBE.LUXIDOOR.ordinal()] = 2;
        for (Types.TRIBE tribe : Types.TRIBE.values())
            assertEq(modern ? expected2015[tribe.ordinal()] : 5, TribesConfig.initialStars(tribe), "starting stars of " + tribe);
        assertEq(modern ? Types.TECHNOLOGY.PHILOSOPHY : Types.TECHNOLOGY.MEDITATION, Types.TRIBE.AI_MO.getInitialTech(), "Ai-Mo tech");
        assertEq(modern ? Types.UNIT.MIND_BENDER : Types.UNIT.WARRIOR, Types.TRIBE.AI_MO.getStartingUnit(), "Ai-Mo unit");
        assertEq(modern ? "STRATEGY" : "SHIELDS", Types.TECHNOLOGY.SHIELDS.getDisplayName(), "Shields display name");
        // Bardur (Phase 1) keeps 5 stars in a real game.
        GameState gs = freshGame();
        assertEq(5, gs.getTribe(0).getStars(), "Bardur starts with 5 stars");
    }

    // ----- Diplomacy (2.15 rules) -----

    private static GameState duelGame() {
        PythonEnv env = new PythonEnv();
        env.initFromLevel(duelLevelFile, 1234L, Types.GAME_MODE.CAPITALS);
        GameState gs = env.getGameState();
        for (int i = 0; i < 2; i++) {
            gs.getTribe(i).getTechTree().doResearchInit(Types.TECHNOLOGY.CLIMBING);
            int other = 1 - i;
            if (!gs.getTribe(i).getTribesMet().contains(other)) gs.getTribe(i).getTribesMet().add(other);
            for (boolean[] row : gs.getTribe(i).getObsGrid()) java.util.Arrays.fill(row, true);
        }
        return gs;
    }

    private static Vector2d freeLandNextTo(GameState gs, Vector2d p) {
        Board b = gs.getBoard();
        for (Vector2d n : p.neighborhood(1, 0, b.getSize()))
            if (b.getUnitAt(n.x, n.y) == null && !b.getTerrainAt(n.x, n.y).isWater() && b.getTerrainAt(n.x, n.y) != Types.TERRAIN.CITY)
                return n;
        throw new AssertionError("no free land tile next to " + p);
    }

    private static Unit addUnit(GameState gs, int tribe, Types.UNIT type, Vector2d pos) {
        City home = (City) gs.getActor(gs.getTribe(tribe).getCapitalID());
        Unit u = Types.UNIT.createUnit(pos.copy(), 0, false, -1, tribe, type);
        gs.getBoard().addUnit(home, u);
        u.setStatus(Types.TURN_STATUS.FRESH);
        return u;
    }

    private static core.actions.tribeactions.DiplomacyAction dip(Types.ACTION type, int me, int other) {
        return new core.actions.tribeactions.DiplomacyAction(type, me, other);
    }

    private static boolean exec(GameState gs, core.actions.Action a) {
        return a.getActionType().getCommand().execute(a, gs);
    }

    private static void testDiplomacyAbsentUnder2020(boolean modern) {
        assertEq(modern, Types.TECHNOLOGY.DIPLOMACY.isActive(), "Diplomacy tech exists");
        assertEq(modern, Types.UNIT.CLOAK.spawnable(), "Cloak trainable");
        assertEq(false, Types.UNIT.DAGGER.spawnable(), "Dagger never trainable");
        GameState gs = duelGame();
        Tribe t = gs.getTribe(0);
        t.getTechTree().doResearchInit(Types.TECHNOLOGY.ORGANIZATION);
        t.getTechTree().doResearchInit(Types.TECHNOLOGY.SHIELDS);
        t.getTechTree().doResearchInit(Types.TECHNOLOGY.DIPLOMACY);
        t.setStars(50);
        int n = new core.actions.tribeactions.factory.DiplomacyFactory().computeActionVariants(t, gs).size();
        assertEq(modern, n > 0, "Diplomacy actions offered");
    }

    private static void testPeaceAndEmbassies() {
        GameState gs = duelGame();
        Board b = gs.getBoard();
        TribeRelations rel = b.getRelations();
        Tribe t0 = gs.getTribe(0), t1 = gs.getTribe(1);
        assertEq(false, dip(Types.ACTION.OFFER_PEACE, 0, 1).isFeasible(gs), "peace offer needs Strategy");
        t0.getTechTree().doResearchInit(Types.TECHNOLOGY.ORGANIZATION);
        t0.getTechTree().doResearchInit(Types.TECHNOLOGY.SHIELDS);
        assertEq(true, exec(gs, dip(Types.ACTION.OFFER_PEACE, 0, 1)), "tribe 0 offers peace");
        assertEq(false, dip(Types.ACTION.OFFER_PEACE, 0, 1).isFeasible(gs), "no second offer");
        assertEq(true, exec(gs, dip(Types.ACTION.ACCEPT_PEACE, 1, 0)), "tribe 1 accepts (no Strategy needed)");
        assertEq(true, rel.isAtPeace(0, 1) && rel.isAtPeace(1, 0), "at peace");

        // At peace: no attacks either way.
        City cap1 = (City) gs.getActor(t1.getCapitalID());
        Vector2d spot = freeLandNextTo(gs, cap1.getPosition());
        Unit w0 = addUnit(gs, 0, Types.UNIT.WARRIOR, spot);
        Unit w1 = addUnit(gs, 1, Types.UNIT.WARRIOR, freeLandNextTo(gs, spot));
        core.actions.unitactions.Attack att = new core.actions.unitactions.Attack(w0.getActorId());
        att.setTargetId(w1.getActorId());
        assertEq(false, att.isFeasible(gs), "no attack at peace");

        // Embassy: Diplomacy, 5 stars, +4 each at peace.
        t0.getTechTree().doResearchInit(Types.TECHNOLOGY.DIPLOMACY);
        assertEq(4, rel.embassyCost(0), "first embassy costs 4 stars");
        t0.setStars(3);
        assertEq(false, dip(Types.ACTION.BUILD_EMBASSY, 0, 1).isFeasible(gs), "embassy needs 4 stars");
        t0.setStars(4);
        assertEq(true, exec(gs, dip(Types.ACTION.BUILD_EMBASSY, 0, 1)), "embassy built");
        assertEq(0, t0.getStars(), "embassy cost");
        assertEq(6, rel.embassyCost(0), "the next embassy costs 2 more");
        assertEq(4, rel.embassyCost(1), "the other tribe's first embassy still costs 4");
        assertEq(true, rel.hasEmbassy(0, 1), "embassy recorded");
        assertEq(false, dip(Types.ACTION.BUILD_EMBASSY, 0, 1).isFeasible(gs), "one embassy per tribe");
        assertEq(TribesConfig.EMBASSY_INCOME_AT_PEACE, rel.embassyIncome(0), "owner income at peace");
        assertEq(TribesConfig.EMBASSY_INCOME_AT_PEACE, rel.embassyIncome(1), "host income at peace");

        // Breaking the treaty: tribe 0's units freeze, its units inside tribe 1's borders are disbanded.
        Unit inside = w0;  // next to tribe 1's capital, so inside its borders
        Unit home = addUnit(gs, 0, Types.UNIT.WARRIOR, freeLandNextTo(gs, ((City) gs.getActor(t0.getCapitalID())).getPosition()));
        assertEq(true, exec(gs, dip(Types.ACTION.BREAK_PEACE, 0, 1)), "tribe 0 breaks the treaty");
        assertEq(false, rel.isAtPeace(0, 1), "no longer at peace");
        assertEq(null, gs.getActor(inside.getActorId()), "unit inside the other's borders disbanded");
        assertEq(Types.TURN_STATUS.FINISHED, home.getStatus(), "other units frozen");
        assertEq(false, rel.canBeHostile(0, 1), "no hostility on the turn the treaty is broken");
        assertEq(TribesConfig.EMBASSY_INCOME, rel.embassyIncome(1), "embassy income back to 2");
        rel.startTurn(0);
        assertEq(true, rel.canBeHostile(0, 1), "hostility allowed from the next turn");

        // An attack destroys the embassy and starts a war, which blocks new embassies.
        Unit a0 = addUnit(gs, 0, Types.UNIT.WARRIOR, freeLandNextTo(gs, w1.getPosition()));
        core.actions.unitactions.Attack att2 = new core.actions.unitactions.Attack(a0.getActorId());
        att2.setTargetId(w1.getActorId());
        assertEq(true, exec(gs, att2), "attack after the treaty");
        assertEq(false, rel.hasEmbassy(0, 1), "attack destroys the embassy");
        assertEq(true, rel.isAtWar(0, 1, gs.getTick()), "at war after an attack");
        t0.setStars(10);
        assertEq(false, dip(Types.ACTION.BUILD_EMBASSY, 0, 1).isFeasible(gs), "no embassy while at war");
    }

    private static void testOfferLapses() {
        GameState gs = duelGame();
        TribeRelations rel = gs.getBoard().getRelations();
        gs.getTribe(0).getTechTree().doResearchInit(Types.TECHNOLOGY.ORGANIZATION);
        gs.getTribe(0).getTechTree().doResearchInit(Types.TECHNOLOGY.SHIELDS);
        exec(gs, dip(Types.ACTION.OFFER_PEACE, 0, 1));
        rel.endTurn(0);
        assertEq(true, rel.hasPeaceOffer(0, 1), "offer survives the offerer's turn end");
        rel.endTurn(1);
        assertEq(false, rel.hasPeaceOffer(0, 1), "offer lapses at the end of the recipient's turn");
    }

    private static void testInfiltration() {
        GameState gs = duelGame();
        Board b = gs.getBoard();
        Tribe t0 = gs.getTribe(0), t1 = gs.getTribe(1);
        City cap1 = (City) gs.getActor(t1.getCapitalID());
        Unit defender = b.getUnitAt(cap1.getPosition().x, cap1.getPosition().y);
        Unit cloak = addUnit(gs, 0, Types.UNIT.CLOAK, freeLandNextTo(gs, cap1.getPosition()));
        core.actions.unitactions.Infiltrate inf = new core.actions.unitactions.Infiltrate(cloak.getActorId());
        inf.setTargetCityId(cap1.getActorId());
        assertEq(true, inf.isFeasible(gs), "infiltration feasible next to an enemy city");
        int stars = t0.getStars();
        int income = cap1.getProduction() - cap1.getMarketIncome();
        int unitsBefore = gs.getUnits(0).size();
        int hpBefore = defender == null ? 0 : defender.getCurrentHP();
        assertEq(true, exec(gs, inf), "infiltration executes");
        assertEq(null, gs.getActor(cloak.getActorId()), "the Cloak is consumed");
        assertEq(stars + income, t0.getStars(), "infiltrator takes the city's income");
        assertEq(true, cap1.isInfiltrated(), "city marked infiltrated");
        int daggers = 0;
        for (Unit u : gs.getUnits(0)) if (u.getType() == Types.UNIT.DAGGER) {
            daggers++;
            assertEq(Types.TURN_STATUS.FINISHED, u.getStatus(), "daggers cannot act this turn");
            assertEq(-1, u.getCityId(), "daggers belong to no city");
        }
        int expected = Math.min(cap1.getLevel(), TribesConfig.MAX_DAGGERS_PER_INFILTRATION);
        assertEq(expected, daggers, "dagger count = min(level, 5)");
        assertEq(unitsBefore - 1 + expected, gs.getUnits(0).size(), "units: cloak out, daggers in");
        if (defender != null && gs.getActor(defender.getActorId()) != null)
            assertEq(true, defender.getCurrentHP() < hpBefore, "defender damaged");
        Unit cloak2 = addUnit(gs, 0, Types.UNIT.CLOAK, freeLandNextTo(gs, cap1.getPosition()));
        core.actions.unitactions.Infiltrate inf2 = new core.actions.unitactions.Infiltrate(cloak2.getActorId());
        inf2.setTargetCityId(cap1.getActorId());
        assertEq(false, inf2.isFeasible(gs), "no second infiltration before the owner's turn");
    }

    private static void testHiddenCloak() {
        GameState gs = duelGame();
        Board b = gs.getBoard();
        Vector2d start = freeLandNextTo(gs, ((City) gs.getActor(gs.getTribe(1).getCapitalID())).getPosition());
        Unit cloak = addUnit(gs, 0, Types.UNIT.CLOAK, start);
        ((core.actors.units.Cloak) cloak).setHidden(true);
        Board pov1 = b.copy(true, 1);
        gs.getTribe(1).getObsGrid()[start.x][start.y] = true;
        pov1 = b.copy(true, 1);
        assertEq(null, pov1.getActor(cloak.getActorId()), "hidden Cloak absent from the enemy's view");
        assertEq(null, pov1.getUnitAt(start.x, start.y), "hidden Cloak tile looks empty to the enemy");
        Board pov0 = b.copy(true, 0);
        assertEq(true, pov0.getActor(cloak.getActorId()) != null, "owner still sees its Cloak");

        Unit w1 = addUnit(gs, 1, Types.UNIT.WARRIOR, freeLandNextTo(gs, start));
        core.actions.unitactions.Attack att = new core.actions.unitactions.Attack(w1.getActorId());
        att.setTargetId(cloak.getActorId());
        assertEq(false, att.isFeasible(gs), "hidden Cloak cannot be attacked");
        core.actions.unitactions.Move mv = new core.actions.unitactions.Move(w1.getActorId());
        mv.setDestination(start.copy());
        Vector2d before = w1.getPosition().copy();
        assertEq(true, exec(gs, mv), "moving onto the hidden Cloak is allowed");
        assertEq(before, w1.getPosition(), "the move is cancelled");
        assertEq(false, cloak.isHidden(), "the Cloak is revealed");
        assertEq(Types.TURN_STATUS.FRESH, w1.getStatus(), "the mover keeps its action");
        assertEq(true, att.isFeasible(gs), "a revealed Cloak can be attacked");
    }

    private static void testDaggerSkills() {
        GameState gs = duelGame();
        Vector2d spot = freeLandNextTo(gs, ((City) gs.getActor(gs.getTribe(0).getCapitalID())).getPosition());
        Unit dagger = addUnit(gs, 0, Types.UNIT.DAGGER, spot);
        Unit def = addUnit(gs, 1, Types.UNIT.DEFENDER, freeLandNextTo(gs, spot));
        core.actions.unitactions.Attack att = new core.actions.unitactions.Attack(dagger.getActorId());
        att.setTargetId(def.getActorId());
        assertEq(true, exec(gs, att), "dagger attacks");
        assertEq(dagger.getMaxHP(), dagger.getCurrentHP(), "Surprise: no retaliation");
        dagger.setKills(TribesConfig.VETERAN_KILLS);
        assertEq(false, new core.actions.unitactions.MakeVeteran(dagger.getActorId()).isFeasible(gs), "Static: no veteran");
        // A Cloak never fights and never retaliates.
        Unit cloak = addUnit(gs, 1, Types.UNIT.CLOAK, freeLandNextTo(gs, spot));
        Unit w0 = addUnit(gs, 0, Types.UNIT.WARRIOR, freeLandNextTo(gs, cloak.getPosition()));
        core.actions.unitactions.Attack byCloak = new core.actions.unitactions.Attack(cloak.getActorId());
        byCloak.setTargetId(w0.getActorId());
        assertEq(false, byCloak.isFeasible(gs), "a Cloak cannot attack units");
        core.actions.unitactions.Attack onCloak = new core.actions.unitactions.Attack(w0.getActorId());
        onCloak.setTargetId(cloak.getActorId());
        int hp = w0.getCurrentHP();
        exec(gs, onCloak);
        assertEq(hp, w0.getCurrentHP(), "Stiff: the Cloak does not retaliate");
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
