package core.actions.unitactions.command;

import core.TribesConfig;
import core.Types;
import core.actions.Action;
import core.actions.ActionCommand;
import core.actions.unitactions.Infiltrate;
import core.actors.City;
import core.actors.Tribe;
import core.actors.units.Unit;
import core.game.Board;
import core.game.GameState;
import utils.Pair;
import utils.Vector2d;

import java.util.ArrayList;
import java.util.List;

/**
 * A Cloak infiltrates an adjacent enemy city (2.1 Diplomacy, 2.15 rules only). Fandom Cloak page:
 * - a unit of the city's owner standing in the city takes the damage of a full-health Cloak attack (no retaliation);
 * - the Cloak is consumed;
 * - min(city level, 5) Daggers of the Cloak's tribe spawn inside the city's borders: the city centre first, then
 *   tiles where they get a defence bonus (forest with Archery, mountain with Climbing), then any other land tile
 *   they can stand on. They cannot act until their owner's next turn. (On water the game spawns Pirates; naval
 *   units are not implemented, so water tiles are skipped.);
 * - the Cloak's tribe gains stars equal to the city's income, and the city produces no income on its owner's
 *   next turn (Market income is unaffected). It cannot be infiltrated again until then.
 */
public class InfiltrateCommand implements ActionCommand {

    @Override
    public boolean execute(Action a, GameState gs) {
        Infiltrate action = (Infiltrate) a;
        if (!action.isFeasible(gs)) return false;

        Board b = gs.getBoard();
        Unit cloak = (Unit) gs.getActor(action.getUnitId());
        City city = (City) gs.getActor(action.getTargetCityId());
        int tribeId = cloak.getTribeId();
        int ownerId = city.getTribeId();
        Tribe tribe = gs.getTribe(tribeId);
        Vector2d cityPos = city.getPosition();

        core.DiplomacyRules.recordHostility(gs, tribeId, ownerId);

        // 1. Damage the defender in the city.
        Unit defender = b.getUnitAt(cityPos.x, cityPos.y);
        if (defender != null) {
            Pair<Integer, Integer> results = AttackCommand.computeAttackResults(cloak, defender, gs, true);
            int damage = results.getFirst();
            if (defender.getCurrentHP() <= damage) {
                tribe.addKill();
                gs.killUnit(defender);
            } else {
                defender.setCurrentHP(defender.getCurrentHP() - damage);
            }
        }

        // 2. The Cloak is consumed.
        gs.killUnit(cloak);

        // 3. Steal the city's income for this turn.
        int cityIncome = Math.max(0, city.getProduction() - city.getMarketIncome());
        tribe.addStars(cityIncome);
        city.setInfiltrated(true);

        // 4. Spawn Daggers.
        int nDaggers = Math.min(city.getLevel(), TribesConfig.MAX_DAGGERS_PER_INFILTRATION);
        for (Vector2d pos : daggerTiles(gs, city, tribe, nDaggers)) {
            Unit dagger = Types.UNIT.createUnit(pos.copy(), 0, false, -1, tribeId, Types.UNIT.DAGGER);
            dagger.setStatus(Types.TURN_STATUS.FINISHED);
            b.addUnit(city, dagger);
            tribe.addScore(Types.UNIT.DAGGER.getPoints());
            tribe.clearView(pos.x, pos.y, 1, gs.getRandomGenerator(), b);
            tribe.applyDeferredViewEffects(b);
        }
        return true;
    }

    /** Up to n free tiles in the city's borders, in spawn priority order. */
    static List<Vector2d> daggerTiles(GameState gs, City city, Tribe tribe, int n) {
        Board b = gs.getBoard();
        boolean forestBonus = tribe.getTechTree().isResearched(Types.TECHNOLOGY.ARCHERY);
        boolean mountainBonus = tribe.getTechTree().isResearched(TribesConfig.mountainDefenceTech());
        List<Vector2d> centre = new ArrayList<>(), bonus = new ArrayList<>(), rest = new ArrayList<>();
        for (Vector2d t : b.getCityTiles(city.getActorId())) {
            if (b.getUnitAt(t.x, t.y) != null) continue;
            Types.TERRAIN terrain = b.getTerrainAt(t.x, t.y);
            if (terrain.isWater() || !b.traversable(t.x, t.y, tribe.getTribeId())) continue;
            if (t.equals(city.getPosition())) centre.add(t);
            else if ((terrain == Types.TERRAIN.FOREST && forestBonus) || (terrain == Types.TERRAIN.MOUNTAIN && mountainBonus)) bonus.add(t);
            else rest.add(t);
        }
        List<Vector2d> out = new ArrayList<>(centre);
        out.addAll(bonus);
        out.addAll(rest);
        return out.size() > n ? out.subList(0, n) : out;
    }
}
