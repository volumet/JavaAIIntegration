package playground.lab3_tinynn;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import playground.common.Bench;

import java.util.Arrays;
import java.util.random.RandomGenerator;
import java.util.stream.IntStream;

/**
 * Lab 3.1: matrix multiplication. Running a neural network is mostly this one operation.
 *
 * <p>All matrices are n x n, stored row-major in a flat float[] ({@code m[row * n + col]}).
 * We build up four versions and watch the GFLOP/s climb:
 * <ol>
 *   <li>naive i-j-k: the textbook loop. The inner loop walks down a column of B (cache-hostile).</li>
 *   <li>i-k-j: the same math, reordered so every inner loop streams along rows (cache-friendly,
 *       and the JIT can auto-vectorize it).</li>
 *   <li>Vector API i-k-j: explicit SIMD with fma.</li>
 *   <li>parallel Vector API: rows split across all cores, the CPU version of a "GPU grid".</li>
 * </ol>
 */
public class MatMul {

    static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    public static void naive(float[] a, float[] b, float[] c, int n) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0f;
                for (int k = 0; k < n; k++) {
                    sum += a[i * n + k] * b[k * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }

    public static void ikj(float[] a, float[] b, float[] c, int n) {
        Arrays.fill(c, 0f);
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                float aik = a[i * n + k];
                for (int j = 0; j < n; j++) {
                    c[i * n + j] += aik * b[k * n + j];
                }
            }
        }
    }

    /** Computes one row of C: c[i,:] = sum over k of a[i,k] * b[k,:]. */
    static void vectorRow(float[] a, float[] b, float[] c, int n, int i) {
        int upper = SPECIES.loopBound(n);
        int rowC = i * n;
        Arrays.fill(c, rowC, rowC + n, 0f);
        for (int k = 0; k < n; k++) {
            float aikScalar = a[i * n + k];
            FloatVector aik = FloatVector.broadcast(SPECIES, aikScalar); // same value in every lane
            int rowB = k * n;
            int j = 0;
            for (; j < upper; j += SPECIES.length()) {
                FloatVector vb = FloatVector.fromArray(SPECIES, b, rowB + j);
                FloatVector vc = FloatVector.fromArray(SPECIES, c, rowC + j);
                aik.fma(vb, vc).intoArray(c, rowC + j);
            }
            for (; j < n; j++) {
                c[rowC + j] += aikScalar * b[rowB + j];
            }
        }
    }

    public static void vector(float[] a, float[] b, float[] c, int n) {
        for (int i = 0; i < n; i++) vectorRow(a, b, c, n, i);
    }

    public static void parallelVector(float[] a, float[] b, float[] c, int n) {
        IntStream.range(0, n).parallel().forEach(i -> vectorRow(a, b, c, n, i));
    }

    public static float maxAbsDiff(float[] x, float[] y) {
        float max = 0f;
        for (int i = 0; i < x.length; i++) max = Math.max(max, Math.abs(x[i] - y[i]));
        return max;
    }

    void main(String[] args) {
        int n = args.length > 0 ? Integer.parseInt(args[0]) : 512;
        RandomGenerator rnd = RandomGenerator.getDefault();
        float[] a = new float[n * n];
        float[] b = new float[n * n];
        for (int i = 0; i < a.length; i++) {
            a[i] = rnd.nextFloat() - 0.5f;
            b[i] = rnd.nextFloat() - 0.5f;
        }
        float[] ref = new float[n * n];
        float[] c = new float[n * n];
        double flops = 2.0 * n * n * n;

        IO.println("C = A x B with n = " + n + " (" + (long) flops / 1_000_000 + " MFLOP per multiply)");
        naive(a, b, ref, n);

        report("naive i-j-k", flops, Bench.measure("naive i-j-k", 2, 5, () -> naive(a, b, c, n)));
        report("i-k-j (cache friendly)", flops, Bench.measure("i-k-j", 3, 5, () -> ikj(a, b, c, n)));
        IO.println("    max |diff| vs naive = " + maxAbsDiff(ref, c));
        report("Vector API", flops, Bench.measure("Vector API", 3, 5, () -> vector(a, b, c, n)));
        IO.println("    max |diff| vs naive = " + maxAbsDiff(ref, c) + "  (fma rounds differently; tiny diffs are OK)");
        report("parallel Vector API", flops, Bench.measure("parallel Vector API", 3, 5, () -> parallelVector(a, b, c, n)));
        IO.println("    max |diff| vs naive = " + maxAbsDiff(ref, c));
    }

    private static void report(String label, double flops, double ms) {
        IO.println("    -> %-24s %6.2f GFLOP/s".formatted(label, flops / (ms / 1000) / 1e9));
    }
}
