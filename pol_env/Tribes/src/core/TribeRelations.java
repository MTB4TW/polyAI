package core;

/**
 * Tribe relations of the 2.1 Diplomacy update (2.15 rules only): peace treaties, pending peace offers,
 * embassies and the "at war" state. Lives on the Board and is copied with it. Under the 2020 rules nothing
 * writes to it, so it never changes the game.
 *
 * Game rules (fandom Strategy, Embassy, Tribe Relations pages):
 * - Two tribes are at war if either attacked the other at most a turn ago; otherwise they are neutral, or at
 *   peace if both agreed to a peace treaty. Here "a turn ago" means in the current or the previous round (tick).
 * - Strategy is needed to send a treaty, not to accept one. A pending offer lapses at the end of the
 *   recipient's next turn (the game shows it as rejected).
 * - At peace no hostile action is possible. Breaking a treaty freezes the breaker's units for the rest of the
 *   turn and disbands its units inside the other tribe's borders (BreakPeaceCommand).
 * - An embassy pays EMBASSY_INCOME to both owner and host each turn, doubled at peace. Attacking a tribe destroys
 *   the embassies between the two.
 */
public class TribeRelations {

    private static final int NEVER = Integer.MIN_VALUE / 2;

    private final int n;
    private final boolean[][] peace;        // symmetric
    private final boolean[][] peaceOffer;   // [from][to]
    private final boolean[][] embassy;      // [owner][host]
    private final int[][] lastHostileTick;  // [attacker][victim]
    private final boolean[][] brokePeace;   // [breaker][other], this turn only

    public TribeRelations(int numTribes) {
        n = numTribes;
        peace = new boolean[n][n];
        peaceOffer = new boolean[n][n];
        embassy = new boolean[n][n];
        lastHostileTick = new int[n][n];
        brokePeace = new boolean[n][n];
        for (int i = 0; i < n; i++) java.util.Arrays.fill(lastHostileTick[i], NEVER);
    }

    public int getNumTribes() { return n; }

    private boolean valid(int a, int b) { return a >= 0 && b >= 0 && a < n && b < n && a != b; }

    public boolean isAtPeace(int a, int b) { return valid(a, b) && peace[a][b]; }

    public boolean isAtWar(int a, int b, int tick) {
        if (!valid(a, b)) return false;
        return lastHostileTick[a][b] >= tick - 1 || lastHostileTick[b][a] >= tick - 1;
    }

    public boolean hasPeaceOffer(int from, int to) { return valid(from, to) && peaceOffer[from][to]; }

    public boolean hasEmbassy(int owner, int host) { return valid(owner, host) && embassy[owner][host]; }

    public boolean brokePeaceThisTurn(int breaker, int other) { return valid(breaker, other) && brokePeace[breaker][other]; }

    /** Whether tribe a may act hostile toward tribe b (attack, capture, convert, infiltrate). */
    public boolean canBeHostile(int a, int b) {
        if (!valid(a, b)) return true;
        return !peace[a][b] && !brokePeace[a][b];
    }

    public void offerPeace(int from, int to) { if (valid(from, to)) peaceOffer[from][to] = true; }

    public void acceptPeace(int accepter, int offerer) {
        if (!valid(accepter, offerer)) return;
        peaceOffer[offerer][accepter] = false;
        peaceOffer[accepter][offerer] = false;
        peace[accepter][offerer] = peace[offerer][accepter] = true;
    }

    public void breakPeace(int breaker, int other) {
        if (!valid(breaker, other)) return;
        peace[breaker][other] = peace[other][breaker] = false;
        brokePeace[breaker][other] = true;
    }

    public void addEmbassy(int owner, int host) { if (valid(owner, host)) embassy[owner][host] = true; }

    public void removeEmbassy(int owner, int host) { if (valid(owner, host)) embassy[owner][host] = false; }

    /** Records a hostile act: the two tribes are at war, and the embassies between them are destroyed. */
    public void recordHostility(int attacker, int victim, int tick) {
        if (!valid(attacker, victim)) return;
        lastHostileTick[attacker][victim] = Math.max(lastHostileTick[attacker][victim], tick);
        embassy[attacker][victim] = false;
        embassy[victim][attacker] = false;
        peaceOffer[attacker][victim] = false;
    }

    /** Called at the start of a tribe's turn: clears its per-turn treaty-breaking flags. */
    public void startTurn(int tribeId) {
        if (tribeId < 0 || tribeId >= n) return;
        java.util.Arrays.fill(brokePeace[tribeId], false);
    }

    /** Called at the end of a tribe's turn: offers it received and did not accept lapse. */
    public void endTurn(int tribeId) {
        if (tribeId < 0 || tribeId >= n) return;
        for (int from = 0; from < n; from++) peaceOffer[from][tribeId] = false;
    }

    /** Stars a tribe earns per turn from embassies it owns and hosts. */
    public int embassyIncome(int tribeId) {
        int income = 0;
        for (int j = 0; j < n; j++) {
            if (j == tribeId) continue;
            int perEmbassy = peace[tribeId][j] ? TribesConfig.EMBASSY_INCOME_AT_PEACE : TribesConfig.EMBASSY_INCOME;
            if (embassy[tribeId][j]) income += perEmbassy;
            if (embassy[j][tribeId]) income += perEmbassy;
        }
        return income;
    }

    public TribeRelations copy() {
        TribeRelations c = new TribeRelations(n);
        for (int i = 0; i < n; i++) {
            System.arraycopy(peace[i], 0, c.peace[i], 0, n);
            System.arraycopy(peaceOffer[i], 0, c.peaceOffer[i], 0, n);
            System.arraycopy(embassy[i], 0, c.embassy[i], 0, n);
            System.arraycopy(lastHostileTick[i], 0, c.lastHostileTick[i], 0, n);
            System.arraycopy(brokePeace[i], 0, c.brokePeace[i], 0, n);
        }
        return c;
    }
}
