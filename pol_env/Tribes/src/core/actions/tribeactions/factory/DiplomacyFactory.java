package core.actions.tribeactions.factory;

import core.Types;
import core.actions.Action;
import core.actions.ActionFactory;
import core.actions.tribeactions.DiplomacyAction;
import core.actors.Actor;
import core.actors.Tribe;
import core.game.GameState;

import java.util.LinkedList;

/** Feasible Diplomacy actions of a tribe toward every other tribe (2.15 rules only). */
public class DiplomacyFactory implements ActionFactory {

    private static final Types.ACTION[] TYPES = {
            Types.ACTION.OFFER_PEACE, Types.ACTION.ACCEPT_PEACE, Types.ACTION.BREAK_PEACE, Types.ACTION.BUILD_EMBASSY
    };

    @Override
    public LinkedList<Action> computeActionVariants(final Actor actor, final GameState gs) {
        LinkedList<Action> actions = new LinkedList<>();
        if (!core.RulesVersion.modern()) return actions;
        Tribe tribe = (Tribe) actor;
        for (Types.ACTION type : TYPES) {
            for (int other = 0; other < gs.getTribes().length; other++) {
                DiplomacyAction a = new DiplomacyAction(type, tribe.getTribeId(), other);
                if (a.isFeasible(gs)) actions.add(a);
            }
        }
        return actions;
    }
}
