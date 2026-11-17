package core.actions.cityactions;

import core.TribesConfig;
import core.Types;
import core.actions.Action;
import core.actors.Tribe;
import core.game.Board;
import core.game.GameState;
import core.actors.City;

public class BurnForest extends CityAction
{
    public BurnForest(int cityId) {
        super(Types.ACTION.BURN_FOREST);
        super.cityId = cityId;
    }

    @Override
    public boolean isFeasible(final GameState gs) {
        City city = (City) gs.getActor(this.cityId);

        Board b = gs.getBoard();
        if(b.getTerrainAt(targetPos.x, targetPos.y) != Types.TERRAIN.FOREST) return false;
        if(b.getCityIdAt(targetPos.x, targetPos.y) != this.cityId) return false;
        // 2.15 rules: a forest with a building on it cannot be burnt.
        if(core.RulesVersion.modern() && b.getBuildingAt(targetPos.x, targetPos.y) != null) return false;

        Tribe t = gs.getTribe(city.getTribeId());
        if(t.getStars() < TribesConfig.burnForestCost()) return false;
        return t.getTechTree().isResearched(TribesConfig.burnForestTech());
    }

    @Override
    public Action copy() {
        BurnForest burn = new BurnForest(this.cityId);
        burn.setTargetPos(this.targetPos.copy());
        return burn;
    }

    public String toString()
    {
        return "BURN_FOREST by city " + this.cityId+ " at " + targetPos;
    }
}
