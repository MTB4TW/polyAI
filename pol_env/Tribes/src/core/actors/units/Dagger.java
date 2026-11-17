package core.actors.units;

import core.Types;
import utils.Vector2d;

import static core.TribesConfig.*;

/**
 * Dagger (2.1 Diplomacy, 2.15 rules only). Spawned only by a Cloak's infiltration, never trained.
 * Skills: Dash, Surprise, Independent, Static.
 */
public class Dagger extends Unit
{
    public Dagger(Vector2d pos, int kills, boolean isVeteran, int cityId, int tribeId) {
        super(DAGGER_ATTACK, DAGGER_DEFENCE, DAGGER_MOVEMENT, DAGGER_MAX_HP, DAGGER_RANGE, DAGGER_COST, pos, kills, isVeteran, cityId, tribeId);
    }

    @Override
    public Types.UNIT getType() {
        return Types.UNIT.DAGGER;
    }

    @Override
    public Dagger copy(boolean hideInfo) {
        Dagger c = new Dagger(getPosition(), getKills(), isVeteran(), getCityId(), getTribeId());
        c.setCurrentHP(getCurrentHP());
        c.setMaxHP(getMaxHP());
        c.setActorId(getActorId());
        c.setStatus(getStatus());
        c.setLastDirection(getLastDirX(), getLastDirY());
        return hideInfo ? (Dagger) c.hide() : c;
    }
}
