package playground.common;

/**
 * A deliberately tiny timing helper, good enough to see order-of-magnitude differences.
 *
 * <p>It is NOT a real benchmark harness: no forking, and only crude protection against dead-code
 * elimination. Once you care about exact numbers, move to JMH (https://github.com/openjdk/jmh).
 */
public final class Bench {

    /** Write results here so the JIT can't prove they are unused and delete the work. */
    public static volatile Object sink;

    private Bench() {}

    /** Runs {@code task} {@code warmups} times untimed, then {@code runs} times timed. Returns the best time in ms. */
    public static double measure(String label, int warmups, int runs, Runnable task) {
        for (int i = 0; i < warmups; i++) task.run();
        long best = Long.MAX_VALUE;
        long total = 0;
        for (int i = 0; i < runs; i++) {
            long t0 = System.nanoTime();
            task.run();
            long dt = System.nanoTime() - t0;
            best = Math.min(best, dt);
            total += dt;
        }
        double bestMs = best / 1e6;
        IO.println("  %-34s best %9.3f ms   avg %9.3f ms".formatted(label, bestMs, total / 1e6 / runs));
        return bestMs;
    }
}
