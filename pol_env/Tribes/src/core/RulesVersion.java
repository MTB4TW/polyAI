package core;

/**
 * Which version of the Polytopia rules the engine plays.
 *
 * TRIBES_2020 is the rule set Tribes shipped with (Polytopia as of 2020) and stays the default, so existing
 * checkpoints and results remain reproducible. POLYTOPIA_2_15 applies the rule changes listed in
 * docs/rules-parity.md (Polytopia 2.15.0, the latest balance update as of 2026).
 *
 * The version is read once from the POLYVISION_RULES environment variable, which the JVM inherits from the
 * Python wrapper. An unknown value fails at class initialisation instead of silently falling back.
 */
public enum RulesVersion {
    TRIBES_2020("tribes_2020"),
    POLYTOPIA_2_15("polytopia_2_15");

    public static final String ENV_VAR = "POLYVISION_RULES";

    private final String id;

    RulesVersion(String id) { this.id = id; }

    public String getId() { return id; }

    public static RulesVersion fromId(String id) {
        if (id == null || id.trim().isEmpty()) return TRIBES_2020;
        String normalised = id.trim().toLowerCase();
        for (RulesVersion v : values()) {
            if (v.id.equals(normalised)) return v;
        }
        throw new IllegalArgumentException("Unknown " + ENV_VAR + " value '" + id + "'; expected one of "
                + java.util.Arrays.toString(ids()));
    }

    public static String[] ids() {
        RulesVersion[] vs = values();
        String[] out = new String[vs.length];
        for (int i = 0; i < vs.length; i++) out[i] = vs[i].id;
        return out;
    }

    private static RulesVersion active = fromId(System.getenv(ENV_VAR));

    /** The rule set this JVM plays. */
    public static RulesVersion active() { return active; }

    /** Identifier of the active rule set, for the Python wrapper's consistency check. */
    public static String activeId() { return active.id; }

    /** True when the current-game (2.15) rules are active. */
    public static boolean modern() { return active == POLYTOPIA_2_15; }

    /** Overrides the active rule set. Only for tests; games must not switch rules mid-run. */
    public static void setActiveForTesting(RulesVersion v) { active = v; }
}
