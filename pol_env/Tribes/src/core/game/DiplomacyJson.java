package core.game;

import core.TribeRelations;
import core.TribesConfig;
import core.Types;
import core.actions.Action;
import core.actions.tribeactions.DiplomacyAction;
import core.actions.unitactions.Infiltrate;
import core.actors.Actor;
import core.actors.Tribe;
import org.json.JSONArray;
import org.json.JSONObject;

/**
 * JSON fields the Python wrapper reads for the 2.1 Diplomacy rules (2.15 rules only). Kept out of PythonEnv so
 * that file only gains three calls. Under the 2020 rules none of these fields are written.
 */
public final class DiplomacyJson {

    private DiplomacyJson() {}

    /** Structured fields of INFILTRATE and the tribe Diplomacy actions; no-op for other actions. */
    public static void addActionFields(JSONObject jo, Action a, GameState gs) {
        if (a instanceof Infiltrate) {
            Infiltrate inf = (Infiltrate) a;
            jo.put("unit_id", inf.getUnitId());
            jo.put("target_city_id", inf.getTargetCityId());
            Actor u = gs.getActor(inf.getUnitId());
            if (u != null) {
                jo.put("src_x", u.getPosition().x);
                jo.put("src_y", u.getPosition().y);
            }
            Actor c = gs.getActor(inf.getTargetCityId());
            if (c != null) {
                jo.put("dst_x", c.getPosition().x);
                jo.put("dst_y", c.getPosition().y);
                jo.put("target_x", c.getPosition().x);
                jo.put("target_y", c.getPosition().y);
            }
        } else if (a instanceof DiplomacyAction) {
            DiplomacyAction d = (DiplomacyAction) a;
            jo.put("tribe_id", d.getTribeId());
            jo.put("target_tribe_id", d.getTargetTribeId());
        }
    }

    /** Adds "hidden" for a Cloak (Hide skill). Only own or visible Cloaks reach the observation. */
    public static void addUnitFields(JSONObject uInfo, core.actors.units.Unit u) {
        if (u.getType() == Types.UNIT.CLOAK) uInfo.put("hidden", u.isHidden());
    }

    /** Star cost of a Diplomacy action, or -1 if the action is not one. */
    public static int starCost(Action a, GameState gs) {
        if (a instanceof DiplomacyAction && a.getActionType() == Types.ACTION.BUILD_EMBASSY)
            return gs.getBoard().getRelations().embassyCost(((DiplomacyAction) a).getTribeId());
        if (a instanceof DiplomacyAction || a instanceof Infiltrate) return 0;
        return -1;
    }

    /** Adds the tribe's relations with every tribe (indexed by tribe id) under "diplomacy"; 2.15 rules only. */
    public static void addTribeFields(JSONObject tribeInfo, Tribe t, GameState state) {
        if (!core.RulesVersion.modern()) return;
        TribeRelations rel = state.getBoard().getRelations();
        int me = t.getTribeId();
        int n = rel.getNumTribes();
        JSONArray peace = new JSONArray(), war = new JSONArray(), offerTo = new JSONArray(), offerFrom = new JSONArray(),
                embassyIn = new JSONArray(), embassyFrom = new JSONArray(), brokePeace = new JSONArray();
        for (int j = 0; j < n; j++) {
            peace.put(rel.isAtPeace(me, j));
            war.put(rel.isAtWar(me, j, state.getTick()));
            offerTo.put(rel.hasPeaceOffer(me, j));
            offerFrom.put(rel.hasPeaceOffer(j, me));
            embassyIn.put(rel.hasEmbassy(me, j));
            embassyFrom.put(rel.hasEmbassy(j, me));
            brokePeace.put(rel.brokePeaceThisTurn(me, j));
        }
        JSONObject d = new JSONObject();
        d.put("peace", peace);
        d.put("war", war);
        d.put("offer_to", offerTo);
        d.put("offer_from", offerFrom);
        d.put("embassy_in", embassyIn);
        d.put("embassy_from", embassyFrom);
        d.put("broke_peace", brokePeace);
        tribeInfo.put("diplomacy", d);
    }
}
