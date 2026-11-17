package playground;

import org.junit.jupiter.api.Test;
import playground.lab1_vector.ArrayAddition;
import playground.lab1_vector.DotProduct;
import playground.lab2_ffm.VectorOnSegments;
import playground.lab3_tinynn.MatMul;
import playground.lab3_tinynn.TinyMLP;
import playground.lab4_simt.KernelSimulator;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Random;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Every SIMD / off-heap / kernel version must agree with its plain-Java version.
 * Lengths like 1, 7, and 1003 make sure the tail handling works with any lane count.
 * Write a test like these whenever you add an exercise solution.
 */
class LabsTest {

    static final int[] SIZES = {0, 1, 7, 8, 9, 16, 33, 1003};

    static float[] random(int n, long seed) {
        Random r = new Random(seed);
        float[] a = new float[n];
        for (int i = 0; i < n; i++) a[i] = r.nextFloat() - 0.5f;
        return a;
    }

    @Test
    void arrayAdditionVariantsAgree() {
        for (int n : SIZES) {
            float[] a = random(n, 1), b = random(n, 2);
            float[] expected = new float[n], v = new float[n], m = new float[n];
            ArrayAddition.scalarAdd(a, b, expected);
            ArrayAddition.vectorAdd(a, b, v);
            ArrayAddition.vectorAddMasked(a, b, m);
            assertArrayEquals(expected, v, "vectorAdd n=" + n);
            assertArrayEquals(expected, m, "vectorAddMasked n=" + n);
        }
    }

    @Test
    void dotProductMatchesWithinTolerance() {
        for (int n : SIZES) {
            float[] a = random(n, 3), b = random(n, 4);
            assertEquals(DotProduct.dotScalar(a, b), DotProduct.dotVector(a, b), 1e-3f, "n=" + n);
        }
    }

    @Test
    void offHeapMatchesOnHeap() {
        int n = 1003;
        float[] a = random(n, 5), b = random(n, 6);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment sa = arena.allocateFrom(JAVA_FLOAT, a);
            MemorySegment sb = arena.allocateFrom(JAVA_FLOAT, b);
            MemorySegment sc = arena.allocate(JAVA_FLOAT, n);
            VectorOnSegments.add(sa, sb, sc, n);
            float[] expected = new float[n];
            ArrayAddition.scalarAdd(a, b, expected);
            assertArrayEquals(expected, sc.toArray(JAVA_FLOAT));
            assertEquals(DotProduct.dotScalar(a, b), VectorOnSegments.dot(sa, sb, n), 1e-3f);
        }
    }

    @Test
    void matMulVariantsAgree() {
        for (int n : new int[] {1, 7, 17, 64}) {
            float[] a = random(n * n, 7), b = random(n * n, 8);
            float[] ref = new float[n * n], c = new float[n * n];
            MatMul.naive(a, b, ref, n);
            MatMul.ikj(a, b, c, n);
            assertTrue(MatMul.maxAbsDiff(ref, c) < 1e-4f, "ikj n=" + n);
            MatMul.vector(a, b, c, n);
            assertTrue(MatMul.maxAbsDiff(ref, c) < 1e-4f, "vector n=" + n);
            MatMul.parallelVector(a, b, c, n);
            assertTrue(MatMul.maxAbsDiff(ref, c) < 1e-4f, "parallelVector n=" + n);
        }
    }

    @Test
    void xorNetwork() {
        var hidden = new TinyMLP.Dense(2, 2, new float[] {1, 1, 1, 1}, new float[] {0, -1});
        var output = new TinyMLP.Dense(2, 1, new float[] {1, -2}, new float[] {0});
        float[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        float[] expected = {0, 1, 1, 0};
        for (int i = 0; i < inputs.length; i++) {
            assertEquals(expected[i], output.forward(TinyMLP.relu(hidden.forward(inputs[i])))[0], 0f);
        }
    }

    @Test
    void reluAndSoftmax() {
        float[] v = {-2, -1, 0, 1, 2, -3, 4, -5, 6, 7, -8};
        float[] scalar = TinyMLP.reluScalar(v.clone());
        assertArrayEquals(scalar, TinyMLP.relu(v.clone()));
        float[] p = TinyMLP.softmax(new float[] {1, 2, 3, 1000});
        double sum = 0;
        for (float f : p) sum += f;
        assertEquals(1.0, sum, 1e-5);
    }

    @Test
    void kernelSimulatorCoversEveryIndexExactlyOnce() {
        int n = 1003;
        int[] hits = new int[n];
        KernelSimulator.dispatch(n, 64, kc -> {
            if (kc.x() < kc.maxX()) hits[kc.x()]++;
        });
        assertTrue(Arrays.stream(hits).allMatch(h -> h == 1));
    }
}
