package core.actors.units;

import core.TribesConfig;
import core.Types;
import utils.Vector2d;

import static core.TribesConfig.*;

/**
 * Cloak (2.1 Diplomacy, 2.15 rules only). Skills: Hide, Infiltrate, Dash, Scout, Creep, Stiff, Static.
 * It is visible when trained, after capturing a village or examining a ruin, and when an enemy tries to move
 * onto its tile; it hides again when it moves.
 */
public class Cloak extends Unit
{
    private boolean hidden = false;
    // Hide and Dash conflict: a Cloak can move and then infiltrate only if it began the turn hidden.
    private boolean hiddenAtTurnStart = false;

    public Cloak(Vector2d pos, int kills, boolean isVeteran, int cityId, int tribeId) {
        super(CLOAK_ATTACK, CLOAK_DEFENCE_INT, CLOAK_MOVEMENT, CLOAK_MAX_HP, CLOAK_RANGE, CLOAK_COST, pos, kills, isVeteran, cityId, tribeId);
    }

    @Override
    public Types.UNIT getType() {
        return Types.UNIT.CLOAK;
    }

    @Override
    public double getDefenceStat() { return TribesConfig.CLOAK_DEFENCE; }

    @Override
    public boolean isHidden() { return hidden; }

    public void setHidden(boolean hidden) { this.hidden = hidden; }

    public boolean wasHiddenAtTurnStart() { return hiddenAtTurnStart; }

    public void setHiddenAtTurnStart(boolean h) { this.hiddenAtTurnStart = h; }

    @Override
    public Cloak copy(boolean hideInfo) {
        Cloak c = new Cloak(getPosition(), getKills(), isVeteran(), getCityId(), getTribeId());
        c.setCurrentHP(getCurrentHP());
        c.setMaxHP(getMaxHP());
        c.setActorId(getActorId());
        c.setStatus(getStatus());
        c.setLastDirection(getLastDirX(), getLastDirY());
        c.hidden = hidden;
        c.hiddenAtTurnStart = hiddenAtTurnStart;
        return hideInfo ? (Cloak) c.hide() : c;
    }
}
