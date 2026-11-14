package core;

import core.actors.units.Unit;
import core.game.GameState;

/**
 * Rule helpers for the 2.1 Diplomacy update (2.15 rules only). Every method returns the 2020 behaviour when
 * the 2020 rules are active, so call sites can use them unconditionally.
 */
public final class DiplomacyRules {

    private DiplomacyRules() {}

    public static TribeRelations relations(GameState gs) { return gs.getBoard().getRelations(); }

    /** True if unit u is a hidden Cloak that the observer cannot see (not its owner, not at peace with it). */
    public static boolean isHiddenFrom(GameState gs, Unit u, int observerTribe) {
        if (!RulesVersion.modern() || u == null || !u.isHidden()) return false;
        return u.getTribeId() != observerTribe && !relations(gs).isAtPeace(u.getTribeId(), observerTribe);
    }

    /** Whether tribe a may act hostile toward tribe b. Always true under the 2020 rules. */
    public static boolean canBeHostile(GameState gs, int a, int b) {
        if (!RulesVersion.modern() || a == b) return true;
        return relations(gs).canBeHostile(a, b);
    }

    /** Records a hostile act of tribe a against tribe b (war state, embassies destroyed). No-op under 2020 rules. */
    public static void recordHostility(GameState gs, int a, int b) {
        if (!RulesVersion.modern() || a == b || a < 0 || b < 0) return;
        relations(gs).recordHostility(a, b, gs.getTick());
    }

    /**
     * Whether another tribe's unit stops the mover from entering (or passing through) its tile.
     * 2.15: a unit at peace, a Cloak or Dagger mover (Hide/Creep) and a hidden Cloak the mover cannot see do not.
     */
    public static boolean blocksPassage(GameState gs, Unit mover, Unit other) {
        if (other == null || other.getTribeId() == mover.getTribeId()) return false;
        if (!RulesVersion.modern()) return true;
        if (relations(gs).isAtPeace(mover.getTribeId(), other.getTribeId())) return false;
        if (isHiddenFrom(gs, other, mover.getTribeId())) return false;
        return !(mover.getType().canHide() || mover.getType().hasCreep());
    }

    /** Whether another tribe's unit exerts a zone of control on the mover. */
    public static boolean exertsZoneOfControl(GameState gs, Unit mover, Unit other) {
        if (other == null || other.getTribeId() == mover.getTribeId()) return false;
        if (!RulesVersion.modern()) return true;
        if (mover.getType().canHide() || mover.getType().hasCreep()) return false;
        if (relations(gs).isAtPeace(mover.getTribeId(), other.getTribeId())) return false;
        return !isHiddenFrom(gs, other, mover.getTribeId());
    }

    /** Whether the mover may use a road or city owned by the tribe that controls city cityId (own, neutral or at peace). */
    public static boolean canUseRoadOf(GameState gs, Unit mover, int cityId) {
        if (cityId == -1) return true;
        core.actors.Tribe t = gs.getBoard().getTribe(mover.getTribeId());
        if (t.controlsCity(cityId)) return true;
        if (!RulesVersion.modern()) return false;
        core.actors.Actor c = gs.getActor(cityId);
        return c != null && relations(gs).isAtPeace(mover.getTribeId(), c.getTribeId());
    }

    /** Whether tile (x, y) looks free to the mover: empty, or holding an enemy hidden Cloak it cannot see. */
    public static boolean looksVacant(GameState gs, Unit mover, int x, int y) {
        Unit u = gs.getBoard().getUnitAt(x, y);
        return u == null || isHiddenFrom(gs, u, mover.getTribeId());
    }
}
