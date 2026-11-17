package playground.lab1_vector;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorSpecies;
import playground.common.Bench;

import java.util.Arrays;
import java.util.random.RandomGenerator;

/**
 * Lab 1.1: the article's first example, element-wise array addition.
 *
 * <p>Three versions of {@code c = a + b}:
 * <ol>
 *   <li>plain scalar loop</li>
 *   <li>vector main loop + scalar tail loop (the article's version)</li>
 *   <li>vector loop with a mask for the last partial chunk (no tail loop)</li>
 * </ol>
 */
public class ArrayAddition {

    static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    public static void scalarAdd(float[] a, float[] b, float[] c) {
        for (int i = 0; i < a.length; i++) {
            c[i] = a[i] + b[i];
        }
    }

    public static void vectorAdd(float[] a, float[] b, float[] c) {
        int i = 0;
        int upper = SPECIES.loopBound(a.length); // largest multiple of the lane count <= length
        for (; i < upper; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, i);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
            va.add(vb).intoArray(c, i);
        }
        for (; i < a.length; i++) { // tail: whatever is left over
            c[i] = a[i] + b[i];
        }
    }

    public static void vectorAddMasked(float[] a, float[] b, float[] c) {
        for (int i = 0; i < a.length; i += SPECIES.length()) {
            VectorMask<Float> m = SPECIES.indexInRange(i, a.length); // all-true except in the last chunk
            FloatVector va = FloatVector.fromArray(SPECIES, a, i, m);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, i, m);
            va.add(vb).intoArray(c, i, m);
        }
    }

    void main() {
        int n = 10_000_003; // deliberately NOT a multiple of the lane count, so the tail logic runs
        RandomGenerator rnd = RandomGenerator.getDefault();
        float[] a = new float[n];
        float[] b = new float[n];
        for (int i = 0; i < n; i++) {
            a[i] = rnd.nextFloat();
            b[i] = rnd.nextFloat();
        }
        float[] c1 = new float[n];
        float[] c2 = new float[n];
        float[] c3 = new float[n];

        IO.println("Species " + SPECIES + ": " + SPECIES.length() + " floats per vector op, n = " + n);
        scalarAdd(a, b, c1);
        vectorAdd(a, b, c2);
        vectorAddMasked(a, b, c3);
        IO.println("vector == scalar: " + Arrays.equals(c1, c2) + ", masked == scalar: " + Arrays.equals(c1, c3));

        Bench.measure("scalar loop", 20, 20, () -> scalarAdd(a, b, c1));
        Bench.measure("vector + scalar tail", 20, 20, () -> vectorAdd(a, b, c2));
        Bench.measure("vector + mask", 20, 20, () -> vectorAddMasked(a, b, c3));

        IO.println("""

                Expect the numbers to be CLOSE. Here is why:
                  * HotSpot's JIT auto-vectorizes simple loops like scalarAdd on its own.
                  * Adding two 40 MB arrays is limited by memory bandwidth, not by compute.
                The Vector API pays off where auto-vectorization gives up. Run DotProduct next.""");
    }
}
