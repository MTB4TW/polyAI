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
 *   java -cp out:lib/json.jar:<test classes> core.RulesParityTest levels/.../map.csv
 */
public final class RulesParityTest {

    private static String levelFile;
    private static int checks = 0;

    private RulesParityTest() {}

    public static void main(String[] args) {
        if (args.length < 1) throw new IllegalArgumentException("usage: RulesParityTest <phase1 level csv>");
        levelFile = args[0];
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
            Vector2d pos = explorer.getPosition();
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
            if (added == null || t.getStars() != stars || t.getTechTree().getNumResearched() != techs) continue;
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
