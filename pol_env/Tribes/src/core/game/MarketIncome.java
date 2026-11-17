package core.game;

import core.TribesConfig;
import core.Types;
import core.actors.Building;
import core.actors.City;
import core.actors.Tribe;
import utils.Vector2d;

/**
 * Market income (2.8.5 rule, 2.15 rules only). The Market takes the Customs House's place (same building slot,
 * Trade, 5 stars, plain). It must stand next to a Sawmill, Windmill or Forge, and earns 1 star per turn for each
 * level of each adjacent Sawmill, Windmill or Forge of its tribe, up to TribesConfig.MARKET_MAX_INCOME_2_15.
 * A processing building's level is the number of adjacent base buildings it processes (Lumber Huts, Farms, Mines).
 * The income is cached on each city (City.getMarketIncome) and counted in its production.
 */
public final class MarketIncome {

    private MarketIncome() {}

    public static boolean isProcessing(Types.BUILDING b) {
        return b == Types.BUILDING.SAWMILL || b == Types.BUILDING.WINDMILL || b == Types.BUILDING.FORGE;
    }

    /** Recomputes the market income of every city of the tribe. */
    public static void refresh(GameState gs, int tribeId) {
        if (!core.RulesVersion.modern() || tribeId < 0) return;
        Board board = gs.getBoard();
        Tribe tribe = gs.getTribe(tribeId);
        for (int cityId : tribe.getCitiesID()) {
            City city = (City) gs.getActor(cityId);
            if (city == null) continue;
            int income = 0;
            for (Building b : city.getBuildings()) {
                if (b.type == Types.BUILDING.CUSTOMS_HOUSE) income += marketIncomeAt(board, tribe, b.position);
            }
            city.setMarketIncome(income);
        }
    }

    static int marketIncomeAt(Board board, Tribe tribe, Vector2d pos) {
        int total = 0;
        for (Vector2d adj : pos.neighborhood(1, 0, board.getSize())) {
            Types.BUILDING b = board.getBuildingAt(adj.x, adj.y);
            if (!isProcessing(b) || !tribe.controlsCity(board.getCityIdAt(adj.x, adj.y))) continue;
            total += processingLevel(board, tribe, adj, b);
        }
        return Math.min(total, TribesConfig.MARKET_MAX_INCOME_2_15);
    }

    static int processingLevel(Board board, Tribe tribe, Vector2d pos, Types.BUILDING processing) {
        Types.BUILDING base = processing.getMatchingBuilding();
        int level = 0;
        for (Vector2d adj : pos.neighborhood(1, 0, board.getSize())) {
            if (board.getBuildingAt(adj.x, adj.y) == base && tribe.controlsCity(board.getCityIdAt(adj.x, adj.y))) level++;
        }
        return level;
    }
}
