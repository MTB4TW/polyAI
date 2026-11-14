package core.actions.unitactions.command;

import core.Diplomacy;
import core.Types;
import core.actions.Action;
import core.actions.ActionCommand;
import core.actions.unitactions.Capture;
import core.actors.City;
import core.actors.Tribe;
import core.actors.units.Unit;
import core.game.Board;
import core.game.GameState;
import utils.Vector2d;

import static core.TribesConfig.CAPTURE_REPERCUSSION;

public class CaptureCommand implements ActionCommand {

    @Override
    public boolean execute(Action a, GameState gs) {
        Capture action = (Capture) a;
        //Check if action is feasible before execution
        if (action.isFeasible(gs)) {
            Types.TERRAIN captureType = action.getCaptureType();
            int unitId = action.getUnitId();
            int targetCityId = action.getTargetCity();
            // Change city tribe id to execute action
            Unit unit = (Unit) gs.getActor(unitId);
            Board b = gs.getBoard();
            Tribe thisTribe = b.getTribe(unit.getTribeId());

            if (captureType == Types.TERRAIN.CITY) {
                City targetCity = (City) gs.getActor(targetCityId);
                Tribe targetTribe = b.getTribe(targetCity.getTribeId());

                //Update scores
                targetTribe.subtractScore(targetCity.getPointsWorth());
                thisTribe.addScore(targetCity.getPointsWorth());

                //Updating diplomacy
                Diplomacy d = gs.getBoard().getDiplomacy();
                // Updating the relationship of the tribes, deducting 30 for a capture
                d.updateAllegiance(CAPTURE_REPERCUSSION, thisTribe.getTribeId(), targetTribe.getTribeId());
                // Checks consequences of the update
                d.checkConsequences(CAPTURE_REPERCUSSION, thisTribe.getTribeId(), targetTribe.getTribeId());

                //the unit that captures exhausts their turn
                unit.setStatus(Types.TURN_STATUS.FINISHED);

                // 2.15 rules: a hostile act. Embassies hosted in a captured capital are lost (an embassy needs
                // the capital to be held by its original owner).
                core.DiplomacyRules.recordHostility(gs, thisTribe.getTribeId(), targetTribe.getTribeId());
                if (core.RulesVersion.modern() && targetCity.isCapital()) {
                    core.TribeRelations rel = b.getRelations();
                    for (int owner = 0; owner < rel.getNumTribes(); owner++) rel.removeEmbassy(owner, targetTribe.getTribeId());
                }

                boolean captured = b.capture(gs, thisTribe, targetCity.getPosition().x, targetCity.getPosition().y);
                if (core.RulesVersion.modern()) {
                    core.game.MarketIncome.refresh(gs, thisTribe.getTribeId());
                    core.game.MarketIncome.refresh(gs, targetTribe.getTribeId());
                }
                return captured;
            } else if (captureType == Types.TERRAIN.VILLAGE) {
                //the unit that captures exhausts their turn
                unit.setStatus(Types.TURN_STATUS.FINISHED);
                // 2.15 rules: capturing a village reveals a Cloak.
                if (unit instanceof core.actors.units.Cloak) ((core.actors.units.Cloak) unit).setHidden(false);

                Vector2d unitPos = unit.getPosition();
                return b.capture(gs, thisTribe, unitPos.x, unitPos.y);
            }

        }
        return false;
    }
}
