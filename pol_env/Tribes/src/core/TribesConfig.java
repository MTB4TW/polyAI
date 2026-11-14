package core;

public class TribesConfig
{
    /**
     * High-level terrain-generation profiles used by the procedural map generator.
     * These values control the macro land/water shape before detailed terrain/resource passes.
     */
    public enum MAP_TYPE {
        DRYLANDS(1.00, 3, 4),
        LAKES(0.72, 3, 4),
        CONTINENTS(0.55, 3, 4),
        PANGEA(0.50, 4, 5),
        ARCHIPELAGO(0.35, 3, 3),
        WATERWORLD(0.10, 2, 2);

        private final double initialLand;
        private final int smoothing;
        private final int relief;

        MAP_TYPE(double initialLand, int smoothing, int relief) {
            this.initialLand = initialLand;
            this.smoothing = smoothing;
            this.relief = relief;
        }

        public double getInitialLand() {
            return initialLand;
        }

        public int getSmoothing() {
            return smoothing;
        }

        public int getRelief() {
            return relief;
        }
    }

    public static final MAP_TYPE DEFAULT_MAP_TYPE = MAP_TYPE.CONTINENTS;

    /* UNITS */

    // Warrior
    public static final int WARRIOR_ATTACK = 2;
    public static final int WARRIOR_DEFENCE = 2;
    public static final int WARRIOR_MOVEMENT = 1;
    public static final int WARRIOR_MAX_HP = 10;
    public static final int WARRIOR_RANGE = 1;
    public static final int WARRIOR_COST = 2;
    static final int WARRIOR_POINTS = 10;

    //Archer
    public static final int ARCHER_ATTACK = 2;
    public static final int ARCHER_DEFENCE = 1;
    public static final int ARCHER_MOVEMENT = 1;
    public static final int ARCHER_MAX_HP = 10;
    public static final int ARCHER_RANGE = 2;
    public static final int ARCHER_COST = 3;
    static final int ARCHER_POINTS = 15;

    //Catapult
    public static final int CATAPULT_ATTACK = 4;
    public static final int CATAPULT_DEFENCE = 0;
    public static final int CATAPULT_MOVEMENT = 1;
    public static final int CATAPULT_MAX_HP = 10;
    public static final int CATAPULT_RANGE = 3;
    public static final int CATAPULT_COST = 8;
    static final int CATAPULT_POINTS = 40;

    //Swordsman
    public static final int SWORDMAN_ATTACK = 3;
    public static final int SWORDMAN_DEFENCE = 3;
    public static final int SWORDMAN_MOVEMENT = 1;
    public static final int SWORDMAN_MAX_HP = 15;
    public static final int SWORDMAN_RANGE = 1;
    public static final int SWORDMAN_COST = 5;
    static final int SWORDMAN_POINTS = 25;

    //MindBender
    public static final int MINDBENDER_ATTACK = 0;
    public static final int MINDBENDER_DEFENCE = 1;
    public static final int MINDBENDER_MOVEMENT = 1;
    public static final int MINDBENDER_MAX_HP = 10;
    public static final int MINDBENDER_RANGE = 1;
    public static final int MINDBENDER_COST = 5;
    public static final int MINDBENDER_HEAL = 4;
    static final int MINDBENDER_POINTS = 25;

    //Defender
    public static final int DEFENDER_ATTACK = 1;
    public static final int DEFENDER_DEFENCE = 3;
    public static final int DEFENDER_MOVEMENT = 1;
    public static final int DEFENDER_MAX_HP = 15;
    public static final int DEFENDER_RANGE = 1;
    public static final int DEFENDER_COST = 3;
    static final int DEFENDER_POINTS = 15;

    //Knight
    public static final int KNIGHT_ATTACK = 4;
    public static final int KNIGHT_DEFENCE = 1;
    public static final int KNIGHT_MOVEMENT = 3;
    public static final int KNIGHT_MAX_HP = 15;
    public static final int KNIGHT_RANGE = 1;
    public static final int KNIGHT_COST = 8;
    static final int KNIGHT_POINTS = 40;

    //Rider
    public static final int RIDER_ATTACK = 2;
    public static final int RIDER_DEFENCE = 1;
    public static final int RIDER_MOVEMENT = 2;
    public static final int RIDER_MAX_HP = 10;
    public static final int RIDER_RANGE = 1;
    public static final int RIDER_COST = 3;
    static final int RIDER_POINTS = 15;

    // Boat
    public static final int BOAT_ATTACK = 1;
    public static final int BOAT_DEFENCE = 1;
    public static final int BOAT_MOVEMENT = 2;
    public static final int BOAT_RANGE = 2;
    public static final int BOAT_COST = 0;
    static final int BOAT_POINTS = 0;

    // Ship
    public static final int SHIP_ATTACK = 2;
    public static final int SHIP_DEFENCE = 2;
    public static final int SHIP_MOVEMENT = 3;
    public static final int SHIP_RANGE = 2;
    public static final int SHIP_COST = 5;
    static final int SHIP_POINTS = 0;

    // Battleship
    public static final int BATTLESHIP_ATTACK = 4;
    public static final int BATTLESHIP_DEFENCE = 3;
    public static final int BATTLESHIP_MOVEMENT = 3;
    public static final int BATTLESHIP_RANGE = 2;
    public static final int BATTLESHIP_COST = 15;
    static final int BATTLESHIP_POINTS = 0;

    //Superunit
    public static final int SUPERUNIT_ATTACK = 5;
    public static final int SUPERUNIT_DEFENCE = 4;
    public static final int SUPERUNIT_MOVEMENT = 1;
    public static final int SUPERUNIT_MAX_HP = 40;
    public static final int SUPERUNIT_RANGE = 1;
    public static final int SUPERUNIT_COST = 10; //Useful for when unit is disbanded.
    static final int SUPERUNIT_POINTS = 50;

    //Explorer
    public static final int NUM_STEPS = 15;

    //General Unit constants
    public static final double ATTACK_MODIFIER = 4.5;
    public static final double DEFENCE_BONUS = 1.5;
    public static final double DEFENCE_IN_WALLS = 4.0;
    public static final int VETERAN_KILLS = 3;
    public static final int VETERAN_PLUS_HP = 5;
    public static final int RECOVER_PLUS_HP = 2;
    public static final int RECOVER_IN_BORDERS_PLUS_HP = 2;

    /* BUILDINGS */

    // Farm
    static final int FARM_COST = 5;
    static final int FARM_BONUS = 2;
    static final Types.RESOURCE FARM_RES_CONSTRAINT = Types.RESOURCE.CROPS;

    //WindMill
    static final int WIND_MILL_COST = 5;
    static final int WIND_MILL_BONUS = 1;

    // LumberHut
    static final int LUMBER_HUT_COST = 2;
    static final int LUMBER_HUT_BONUS = 1;

    // SawMill
    static final int SAW_MILL_COST = 5;
    static final int SAW_MILL_BONUS = 1;

    //Mine
    static final int MINE_COST = 5;
    static final int MINE_BONUS = 2;
    static final Types.RESOURCE MINE_RES_CONSTRAINT = Types.RESOURCE.ORE;

    // Forge
    static final int FORGE_COST = 5;
    static final int FORGE_BONUS = 2;

    // Port
    static final int PORT_COST = 10;
    static final int PORT_BONUS = 2;
    public static final int PORT_TRADE_DISTANCE = 4; //Count includes destination port.

    // Customs House
    static final int CUSTOMS_COST = 5;
    static final int CUSTOMS_BONUS = 2;

    //Monuments
    static final int MONUMENT_BONUS = 3;
    public static final int MONUMENT_POINTS = 400;
    public static final int EMPERORS_TOMB_STARS = 100;
    public static final int GATE_OF_POWER_KILLS = 10;
    public static final int GRAND_BAZAR_CITIES = 5;
    public static final int ALTAR_OF_PEACE_TURNS = 5;
    public static final int PARK_OF_FORTUNE_LEVEL = 5;


    // Temple
    static final int TEMPLE_COST = 20;
    static final int TEMPLE_FOREST_COST = 15;
    static final int TEMPLE_BONUS = 1;
    public static final int TEMPLE_TURNS_TO_SCORE = 3;
    public static final int[] TEMPLE_POINTS = new int[]{100, 50, 50, 50, 150};

    //Resources
    static final int ANIMAL_COST = 2;
    static final int FISH_COST = 2;
    static final int WHALES_COST = 0;
    static final int FRUIT_COST = 2;
    static final int ANIMAL_POP = 1;
    static final int FISH_POP = 1;
    static final int WHALES_STARS = 10;
    static final int FRUIT_POP = 1;

    // ROAD
    public static final int ROAD_COST = 2;

    // City
    public static final int CITY_LEVEL_UP_WORKSHOP_PROD = 1;
    public static final int CITY_LEVEL_UP_RESOURCES = 5;
    public static final int CITY_LEVEL_UP_POP_GROWTH = 3;
    public static final int CITY_LEVEL_UP_PARK = 250;
    public static final int CITY_BORDER_POINTS = 20;
    public static final int CITY_CENTRE_POINTS = 100;
    public static final int PROD_CAPITAL_BONUS = 1;
    public static final int EXPLORER_CLEAR_RANGE = 1;
    public static final int FIRST_CITY_CLEAR_RANGE = 2;
    public static final int NEW_CITY_CLEAR_RANGE = 1;
    public static final int CITY_EXPANSION_TILES = 1;
    public static final int POINTS_PER_POPULATION = 5;

    // Diplomacy
    public static final int ALLEGIANCE_MAX = 60;
    public static final int ATTACK_REPERCUSSION = -5;
    public static final int CAPTURE_REPERCUSSION = -30;
    public static final int CONVERT_REPERCUSSION = -5;
    public static final int MIN_STARS_SEND = 15;


    // Research
    public static final int TECH_BASE_COST = 4;
    public static final Types.TECHNOLOGY TECH_DISCOUNT = Types.TECHNOLOGY.PHILOSOPHY;
    public static final double TECH_DISCOUNT_VALUE = 0.2;
    public static final int TECH_TIER_POINTS = 100;

    /* TRIBES */
    public static final int INITIAL_STARS = 5;//1000;

    /* ACTIONS */
    public static final int CLEAR_FOREST_STAR = 2;
    public static final int GROW_FOREST_COST = 5;
    public static final int BURN_FOREST_COST = 5;
    public static final int CLEAR_VIEW_POINTS = 5;

    // MAP
    public static final int[] DEFAULT_MAP_SIZE = new int[]{-1, 11, 14, 16, 18, 20, 22, 24};

    /* RULES-VERSION DEPENDENT VALUES
     * The constants above are the TRIBES_2020 rules. Code that applies a rule which differs in
     * POLYTOPIA_2_15 goes through these accessors (see RulesVersion and docs/rules-parity.md). */

    public static final int LUMBER_HUT_COST_2_15 = 3;           // 2.8.0 Path of the Ocean
    public static final int CLEAR_FOREST_STAR_2_15 = 1;         // 2.2
    public static final int ROAD_COST_2_15 = 3;                 // 2.8.0
    public static final int BURN_FOREST_COST_2_15 = 3;          // 2.14.0
    public static final int KNIGHT_MAX_HP_2_15 = 10;            // 2.2
    public static final int CITY_LEVEL_UP_PARK_PROD_2_15 = 1;   // 2.8.0: Park gives +1 star per turn
    public static final int EXPLORER_STEPS_2_15 = 12;           // UNVERIFIED: fandom only (Feb 2026)
    public static final int MEETING_SECOND_PLAYER_BONUS_2_15 = 2; // UNVERIFIED: reported from play, no source

    /** Steps an explorer takes: 15 in 2020; 12 under 2.15 rules (unverified). */
    public static int explorerSteps() {
        return RulesVersion.modern() ? EXPLORER_STEPS_2_15 : NUM_STEPS;
    }

    public static int clearForestStars() {
        return RulesVersion.modern() ? CLEAR_FOREST_STAR_2_15 : CLEAR_FOREST_STAR;
    }

    public static int roadCost() {
        return RulesVersion.modern() ? ROAD_COST_2_15 : ROAD_COST;
    }

    public static int burnForestCost() {
        return RulesVersion.modern() ? BURN_FOREST_COST_2_15 : BURN_FOREST_COST;
    }

    public static int knightMaxHp() {
        return RulesVersion.modern() ? KNIGHT_MAX_HP_2_15 : KNIGHT_MAX_HP;
    }

    /** Technology needed to burn a forest: Chivalry in 2020, Construction since 2.8.5. */
    public static Types.TECHNOLOGY burnForestTech() {
        return RulesVersion.modern() ? Types.TECHNOLOGY.CONSTRUCTION : Types.TECHNOLOGY.CHIVALRY;
    }

    /** Technology needed to destroy a building: Construction in 2020, Chivalry since 2.8.5. */
    public static Types.TECHNOLOGY destroyTech() {
        return RulesVersion.modern() ? Types.TECHNOLOGY.CHIVALRY : Types.TECHNOLOGY.CONSTRUCTION;
    }

    /** Technology that gives the mountain defence bonus: Meditation in 2020, Climbing since 2.0.58. */
    public static Types.TECHNOLOGY mountainDefenceTech() {
        return RulesVersion.modern() ? Types.TECHNOLOGY.CLIMBING : Types.TECHNOLOGY.MEDITATION;
    }

    /**
     * Applies the Literacy (Philosophy) research discount.
     * TRIBES_2020 keeps Tribes' original arithmetic, which charges 20% of the cost (Tribes meant "20% off").
     * POLYTOPIA_2_15 takes 33% off, rounded up: ceil(2 * cost / 3), which reproduces the cost table on the
     * fandom Technology page (2.0.44 onwards).
     */
    public static int applyLiteracyDiscount(int cost) {
        if (RulesVersion.modern()) {
            return (2 * cost + 2) / 3;
        }
        return (int) (cost * TECH_DISCOUNT_VALUE);
    }

    // ---- Step 2 and the remaining 2.15 rules (docs/rules-parity.md) ----
    public static final int MARKET_MAX_INCOME_2_15 = 8;          // 2.8.5: +1 star per level of adjacent Sawmill/Windmill/Forge, max 8
    // 2.14.0 starting stars. Every tribe not listed starts with INITIAL_STARS (5).
    public static final int XIN_XI_STARS_2_15 = 7;
    public static final int OUMAJI_STARS_2_15 = 6;
    public static final int HOODRICK_STARS_2_15 = 7;
    public static final int QUETZALI_STARS_2_15 = 7;
    public static final int YADAKK_STARS_2_15 = 7;
    public static final int LUXIDOOR_STARS_2_15 = 2;             // plus a level-3 capital (already Tribes' Luxidoor rule)

    // Diplomacy (2.1/2.2; Strategy, Diplomacy tech). Fandom Cloak/Dagger/Embassy/Strategy pages, read Nov 2026.
    public static final int CLOAK_ATTACK = 2;
    public static final double CLOAK_DEFENCE = 0.5;
    public static final int CLOAK_DEFENCE_INT = 0;               // the integer DEF field; combat uses getDefenceStat()
    public static final int CLOAK_MOVEMENT = 2;
    public static final int CLOAK_MAX_HP = 5;
    public static final int CLOAK_RANGE = 1;
    public static final int CLOAK_COST = 8;
    static final int CLOAK_POINTS = 40;                          // 5 points per cost star, as for other units
    public static final int DAGGER_ATTACK = 2;
    public static final int DAGGER_DEFENCE = 2;
    public static final int DAGGER_MOVEMENT = 1;
    public static final int DAGGER_MAX_HP = 10;
    public static final int DAGGER_RANGE = 1;
    public static final int DAGGER_COST = 2;                     // not trainable; the fandom's "theoretical" cost (disband value)
    static final int DAGGER_POINTS = 10;
    public static final int MAX_DAGGERS_PER_INFILTRATION = 5;    // daggers = min(city level, 5)
    public static final int EMBASSY_BASE_COST = 4;               // 2.14.0: the first embassy costs 4 stars,
    public static final int EMBASSY_COST_STEP = 2;               // and each later one 2 more (reported by Devin)
    public static final int EMBASSY_INCOME = 2;                  // to both sender and host, per turn
    public static final int EMBASSY_INCOME_AT_PEACE = 4;         // doubled while a peace treaty holds
    public static final int CLOAK_VISION_RANGE = 2;              // Scout skill: 5x5

    /** Stars a tribe starts the game with. */
    public static int initialStars(Types.TRIBE tribe) {
        if (!RulesVersion.modern() || tribe == null) return INITIAL_STARS;
        switch (tribe) {
            case XIN_XI: return XIN_XI_STARS_2_15;
            case OUMAJI: return OUMAJI_STARS_2_15;
            case HOODRICK: return HOODRICK_STARS_2_15;
            case QUETZALI: return QUETZALI_STARS_2_15;
            case YADAKK: return YADAKK_STARS_2_15;
            case LUXIDOOR: return LUXIDOOR_STARS_2_15;
            default: return INITIAL_STARS;
        }
    }

}
