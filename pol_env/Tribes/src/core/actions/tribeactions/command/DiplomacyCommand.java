package core.actions.tribeactions.command;

import core.TribeRelations;
import core.TribesConfig;
import core.Types;
import core.actions.Action;
import core.actions.ActionCommand;
import core.actions.tribeactions.DiplomacyAction;
import core.actors.Actor;
import core.actors.Tribe;
import core.actors.units.Unit;
import core.game.Board;
import core.game.GameState;
import utils.Vector2d;

import java.util.ArrayList;

/** Executes OFFER_PEACE, ACCEPT_PEACE, BREAK_PEACE and BUILD_EMBASSY (2.15 rules only). */
public class DiplomacyCommand implements ActionCommand {

    @Override
    public boolean execute(Action a, GameState gs) {
        DiplomacyAction action = (DiplomacyAction) a;
        if (!action.isFeasible(gs)) return false;
        int me = action.getTribeId();
        int other = action.getTargetTribeId();
        Board b = gs.getBoard();
        TribeRelations rel = b.getRelations();
        Tribe tribe = gs.getTribe(me);

        switch (action.getActionType()) {
            case OFFER_PEACE:
                rel.offerPeace(me, other);
                return true;
            case ACCEPT_PEACE:
                rel.acceptPeace(me, other);
                return true;
            case BREAK_PEACE:
                rel.breakPeace(me, other);
                breakPeaceConsequences(gs, me, other);
                return true;
            case BUILD_EMBASSY:
                tribe.subtractStars(TribesConfig.EMBASSY_COST);
                rel.addEmbassy(me, other);
                // The embassy reveals the 3x3 area around the host capital to its owner.
                Actor capital = gs.getActor(b.getCapitalIDs()[other]);
                Vector2d p = capital.getPosition();
                tribe.clearView(p.x, p.y, 1, gs.getRandomGenerator(), b);
                tribe.applyDeferredViewEffects(b);
                return true;
            default:
                return false;
        }
    }

    /**
     * Breaking a treaty freezes all of the breaker's units for the rest of the turn and disbands, without refund,
     * those inside the other tribe's borders.
     */
    static void breakPeaceConsequences(GameState gs, int breaker, int other) {
        Board b = gs.getBoard();
        ArrayList<Unit> units = new ArrayList<>();
        for (Actor act : b.getActors()) {
            if (act instanceof Unit && act.getTribeId() == breaker) units.add((Unit) act);
        }
        for (Unit u : units) {
            Vector2d p = u.getPosition();
            int cityId = b.getCityIdAt(p.x, p.y);
            Actor c = cityId == -1 ? null : gs.getActor(cityId);
            if (c != null && c.getTribeId() == other) gs.killUnit(u);
            else u.setStatus(Types.TURN_STATUS.FINISHED);
        }
    }
}
