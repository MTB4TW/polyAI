package core.actions.unitactions;

import core.Types;
import core.actions.Action;
import core.actors.City;
import core.actors.units.Cloak;
import core.actors.units.Unit;
import core.game.Board;
import core.game.GameState;
import utils.Vector2d;

/**
 * A Cloak infiltrates an adjacent enemy city (2.1 Diplomacy, 2.15 rules only). See InfiltrateCommand.
 */
public class Infiltrate extends UnitAction
{
    private int targetCityId;

    public Infiltrate(int unitId)
    {
        super(Types.ACTION.INFILTRATE);
        super.unitId = unitId;
    }

    public void setTargetCityId(int cityId) { this.targetCityId = cityId; }
    public int getTargetCityId() { return targetCityId; }

    @Override
    public boolean isFeasible(final GameState gs)
    {
        if (!core.RulesVersion.modern()) return false;
        Unit unit = (Unit) gs.getActor(this.unitId);
        if (!(unit instanceof Cloak)) return false;
        if (!unit.canAttack()) return false;
        // Hide and Dash conflict: after moving, only a Cloak that began the turn hidden can still infiltrate.
        if (unit.getStatus() == Types.TURN_STATUS.MOVED && !((Cloak) unit).wasHiddenAtTurnStart()) return false;

        core.actors.Actor actor = gs.getActor(targetCityId);
        if (!(actor instanceof City)) return false;
        City city = (City) actor;
        int owner = city.getTribeId();
        if (owner == unit.getTribeId()) return false;
        if (!core.DiplomacyRules.canBeHostile(gs, unit.getTribeId(), owner)) return false;
        if (city.isInfiltrated()) return false;

        Board b = gs.getBoard();
        Vector2d cityPos = city.getPosition();
        if (!b.getTribe(unit.getTribeId()).isVisible(cityPos.x, cityPos.y)) return false;
        if (Vector2d.chebychevDistance(unit.getPosition(), cityPos) > 1) return false;

        // A city under siege (another tribe's unit on its centre) cannot be infiltrated.
        Unit inCity = b.getUnitAt(cityPos.x, cityPos.y);
        return inCity == null || inCity.getTribeId() == owner;
    }

    @Override
    public Action copy() {
        Infiltrate c = new Infiltrate(this.unitId);
        c.setTargetCityId(this.targetCityId);
        return c;
    }

    public String toString() { return "INFILTRATE by unit " + this.unitId + " into city " + this.targetCityId; }

    public boolean equals(Object o) {
        if (!(o instanceof Infiltrate)) return false;
        Infiltrate other = (Infiltrate) o;
        return super.equals(other) && targetCityId == other.targetCityId;
    }
}
