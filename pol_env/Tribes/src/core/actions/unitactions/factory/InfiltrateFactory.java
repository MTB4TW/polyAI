package core.actions.unitactions.factory;

import core.actions.Action;
import core.actions.ActionFactory;
import core.actions.unitactions.Infiltrate;
import core.actors.Actor;
import core.actors.City;
import core.actors.units.Cloak;
import core.actors.units.Unit;
import core.game.Board;
import core.game.GameState;
import utils.Vector2d;

import java.util.LinkedList;

/** Infiltrate actions of a Cloak: one per adjacent enemy city centre (2.15 rules only). */
public class InfiltrateFactory implements ActionFactory {

    @Override
    public LinkedList<Action> computeActionVariants(final Actor actor, final GameState gs) {
        LinkedList<Action> actions = new LinkedList<>();
        if (!core.RulesVersion.modern() || !(actor instanceof Cloak)) return actions;
        Unit unit = (Unit) actor;
        Board b = gs.getBoard();
        for (Vector2d tile : unit.getPosition().neighborhood(1, 0, b.getSize())) {
            if (b.getTerrainAt(tile.x, tile.y) != core.Types.TERRAIN.CITY) continue;
            int cityId = b.getCityIdAt(tile.x, tile.y);
            Actor c = gs.getActor(cityId);
            if (!(c instanceof City) || !c.getPosition().equals(tile)) continue;
            Infiltrate a = new Infiltrate(unit.getActorId());
            a.setTargetCityId(cityId);
            if (a.isFeasible(gs)) actions.add(a);
        }
        return actions;
    }
}
