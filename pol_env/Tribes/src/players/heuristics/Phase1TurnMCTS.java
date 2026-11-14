package players.heuristics;

import core.Types;
import core.actions.Action;
import core.game.GameState;

import java.util.ArrayList;
import java.util.Random;

/**
 * Monte Carlo tree search over the action sequence of the CURRENT turn, for Phase 1 (solo, no opponent).
 *
 * END_TURN, a depth cap, or a state with no actions ends a branch; a branch is valued by
 * {@link Phase1SptHeuristic} on the state reached (END_TURN is not applied, so its value is "stop here").
 * Value beyond the current turn comes entirely from the heuristic's forward-looking features.
 *
 * Within a turn the forward model is deterministic, so each node backs up the best leaf value found below it
 * (max-backup) and the root returns the child with the best such value. Selection uses UCB1 on min-max
 * normalised values. Leaves are expanded with a short greedy rollout (one-step heuristic lookahead).
 */
public class Phase1TurnMCTS {

    private static final class Node {
        final Node parent;
        final Action action;     // action that led here (null at root)
        final int depth;
        final boolean terminal;  // END_TURN or depth cap: no children
        ArrayList<Action> actions;
        Node[] children;
        int expanded = 0;
        int visits = 0;
        double best = Double.NEGATIVE_INFINITY;
        boolean dead = false;    // its action could not be applied in the search copy

        Node(Node parent, Action action, int depth, boolean terminal) {
            this.parent = parent;
            this.action = action;
            this.depth = depth;
            this.terminal = terminal;
        }

        void markDead() {
            dead = true;
            actions = new ArrayList<>();
            children = new Node[0];
        }
    }

    private final Phase1SptHeuristic h;
    private final Random rnd;
    private final double c;
    private final int maxDepth;
    private final int rolloutSteps;
    private double vMin = Double.POSITIVE_INFINITY, vMax = Double.NEGATIVE_INFINITY;

    public Phase1TurnMCTS(Phase1SptHeuristic h, long seed, double c, int maxDepth, int rolloutSteps) {
        this.h = h;
        this.rnd = new Random(seed);
        this.c = c;
        this.maxDepth = maxDepth;
        this.rolloutSteps = rolloutSteps;
    }

    /**
     * @param root         state to search from (a copy; it is not modified)
     * @param rootActions  the root's full action list (same order as the live state's getAllAvailableActions())
     * @param allowed      indices into rootActions the caller may choose (the wrapper's legal set)
     * @return index into rootActions of the chosen action
     */
    public int search(GameState root, ArrayList<Action> rootActions, int[] allowed, int iterations) {
        int[] order = allowed.clone();
        for (int i = order.length - 1; i > 0; i--) {
            int j = rnd.nextInt(i + 1);
            int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
        }
        Node rootNode = new Node(null, null, 0, false);
        rootNode.actions = new ArrayList<>();
        for (int i : order) rootNode.actions.add(rootActions.get(i));
        rootNode.children = new Node[rootNode.actions.size()];

        for (int it = 0; it < iterations; it++) {
            GameState s = root.copy();
            Node node = rootNode;
            boolean failed = false;

            // Selection
            while (!node.terminal && node.expanded == node.children.length && node.children.length > 0) {
                node = selectUcb(node);
                if (node.dead) { failed = true; break; }
                if (node.action.getActionType() == Types.ACTION.END_TURN) break;  // terminal: valued without applying
                if (!apply(s, node.action)) { node.markDead(); failed = true; break; }
            }
            if (failed) { backup(node, Double.NEGATIVE_INFINITY); continue; }

            // Expansion
            if (!node.terminal && node.expanded < node.children.length) {
                Action a = node.actions.get(node.expanded);
                boolean endTurn = a.getActionType() == Types.ACTION.END_TURN;
                Node child = new Node(node, a, node.depth + 1, endTurn || node.depth + 1 >= maxDepth);
                node.children[node.expanded++] = child;
                node = child;
                if (!endTurn && !apply(s, a)) { node.markDead(); backup(node, Double.NEGATIVE_INFINITY); continue; }
                if (!node.terminal) {
                    node.actions = new ArrayList<>(s.getAllAvailableActions());
                    node.children = new Node[node.actions.size()];
                    shuffleUntried(node);
                }
            }

            // Evaluation: greedy rollout to the end of the turn (bounded), keeping the best state seen.
            double value = node.terminal ? h.evaluateState(s) : greedyRollout(s);
            backup(node, value);
        }

        int bestIdx = 0;
        double bestVal = Double.NEGATIVE_INFINITY;
        for (int k = 0; k < rootNode.children.length; k++) {
            Node ch = rootNode.children[k];
            if (ch != null && ch.best > bestVal) { bestVal = ch.best; bestIdx = k; }
        }
        return order[bestIdx];
    }

    private boolean apply(GameState s, Action a) {
        try {
            if (!a.isFeasible(s)) return false;
            s.advance(a, true);
            return true;
        } catch (RuntimeException e) {
            return false;
        }
    }

    private double greedyRollout(GameState s) {
        double best = h.evaluateState(s);
        for (int step = 0; step < rolloutSteps; step++) {
            ArrayList<Action> acts = s.getAllAvailableActions();
            GameState bestNext = null;
            double bestNextVal = best;
            for (Action a : acts) {
                if (a.getActionType() == Types.ACTION.END_TURN) continue;
                GameState t = s.copy();
                if (!apply(t, a)) continue;
                double v = h.evaluateState(t);
                if (v > bestNextVal + 1e-9) { bestNextVal = v; bestNext = t; }
            }
            if (bestNext == null) break;  // no improving action: the greedy policy would end the turn
            s = bestNext;
            best = bestNextVal;
        }
        return best;
    }

    private Node selectUcb(Node node) {
        double span = vMax - vMin;
        double logN = Math.log(Math.max(1, node.visits));
        Node pick = null;
        double pickScore = Double.NEGATIVE_INFINITY;
        for (Node ch : node.children) {
            double q = (ch.best == Double.NEGATIVE_INFINITY) ? -1.0 : (span > 0 ? (ch.best - vMin) / span : 0.5);
            double u = q + c * Math.sqrt(logN / Math.max(1, ch.visits)) + 1e-6 * rnd.nextDouble();
            if (u > pickScore) { pickScore = u; pick = ch; }
        }
        return pick;
    }

    private void backup(Node node, double value) {
        if (value != Double.NEGATIVE_INFINITY) {
            vMin = Math.min(vMin, value);
            vMax = Math.max(vMax, value);
        }
        for (Node n = node; n != null; n = n.parent) {
            n.visits++;
            if (value > n.best) n.best = value;
        }
    }

    /** Randomise expansion order so ties and budget limits do not always favour the same actions. */
    private void shuffleUntried(Node node) {
        for (int i = node.actions.size() - 1; i > 0; i--) {
            int j = rnd.nextInt(i + 1);
            Action tmp = node.actions.get(i);
            node.actions.set(i, node.actions.get(j));
            node.actions.set(j, tmp);
        }
    }
}
