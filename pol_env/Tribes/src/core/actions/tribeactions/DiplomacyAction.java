package core.actions.tribeactions;

import core.TribeRelations;
import core.TribesConfig;
import core.Types;
import core.actions.Action;
import core.actors.Actor;
import core.actors.City;
import core.actors.Tribe;
import core.game.GameState;

/**
 * Tribe-level Diplomacy actions toward one other tribe (2.1 Diplomacy, 2.15 rules only):
 * OFFER_PEACE (needs Strategy, the SHIELDS tech), ACCEPT_PEACE (a pending offer from the target),
 * BREAK_PEACE (a treaty with the target) and BUILD_EMBASSY (needs Diplomacy; in the target's capital while it
 * holds it, not at war, one per tribe). All need the target to be a living tribe that this tribe has met.
 */
public class DiplomacyAction extends TribeAction {

    private int targetTribeId;

    public DiplomacyAction(Types.ACTION type, int tribeId, int targetTribeId) {
        super(type);
        this.tribeId = tribeId;
        this.targetTribeId = targetTribeId;
    }

    public int getTargetTribeId() { return targetTribeId; }

    public static boolean isDiplomacyType(Types.ACTION t) {
        return t == Types.ACTION.OFFER_PEACE || t == Types.ACTION.ACCEPT_PEACE
                || t == Types.ACTION.BREAK_PEACE || t == Types.ACTION.BUILD_EMBASSY;
    }

    @Override
    public boolean isFeasible(final GameState gs) {
        if (!core.RulesVersion.modern() || !isDiplomacyType(actionType)) return false;
        Tribe[] tribes = gs.getTribes();
        if (targetTribeId < 0 || targetTribeId >= tribes.length || targetTribeId == tribeId) return false;
        Tribe me = gs.getTribe(tribeId);
        Tribe other = gs.getTribe(targetTribeId);
        if (other.getWinner() == Types.RESULT.LOSS || !me.getTribesMet().contains(targetTribeId)) return false;
        TribeRelations rel = gs.getBoard().getRelations();

        switch (actionType) {
            case OFFER_PEACE:
                return me.getTechTree().isResearched(Types.TECHNOLOGY.SHIELDS)
                        && !rel.isAtPeace(tribeId, targetTribeId)
                        && !rel.hasPeaceOffer(tribeId, targetTribeId)
                        && !rel.hasPeaceOffer(targetTribeId, tribeId)
                        && !rel.brokePeaceThisTurn(tribeId, targetTribeId);
            case ACCEPT_PEACE:
                return rel.hasPeaceOffer(targetTribeId, tribeId) && !rel.isAtPeace(tribeId, targetTribeId);
            case BREAK_PEACE:
                return rel.isAtPeace(tribeId, targetTribeId);
            case BUILD_EMBASSY:
                if (!me.getTechTree().isResearched(Types.TECHNOLOGY.DIPLOMACY)) return false;
                if (me.getStars() < TribesConfig.EMBASSY_COST) return false;
                if (rel.hasEmbassy(tribeId, targetTribeId)) return false;
                if (rel.isAtWar(tribeId, targetTribeId, gs.getTick())) return false;
                Actor capital = gs.getActor(gs.getBoard().getCapitalIDs()[targetTribeId]);
                return capital instanceof City && capital.getTribeId() == targetTribeId;
            default:
                return false;
        }
    }

    @Override
    public Action copy() {
        return new DiplomacyAction(actionType, tribeId, targetTribeId);
    }

    public String toString() {
        return actionType + " by tribe " + tribeId + " toward tribe " + targetTribeId;
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof DiplomacyAction)) return false;
        DiplomacyAction other = (DiplomacyAction) o;
        return super.equals(other) && targetTribeId == other.targetTribeId;
    }
}
