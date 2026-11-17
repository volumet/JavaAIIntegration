package playground.lab2_ffm;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import playground.common.Bench;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.util.random.RandomGenerator;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;

/**
 * Lab 2.4: Vector API + FFM together, SIMD math directly on off-heap memory.
 *
 * <p>This is the combination that pure-Java LLM engines use. The weights stay in a memory-mapped
 * file ({@code FileChannel.map(..., arena)} returns a MemorySegment), and the math runs on them
 * with {@code FloatVector.fromMemorySegment}. Nothing is copied onto the heap.
 */
public class VectorOnSegments {

    static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    static final ByteOrder ORDER = ByteOrder.nativeOrder();

    /** c = a + b, where all three are off-heap float buffers of n elements. */
    public static void add(MemorySegment a, MemorySegment b, MemorySegment c, long n) {
        long lanes = SPECIES.length();
        long upper = n - (n % lanes);
        long i = 0;
        for (; i < upper; i += lanes) {
            long offset = i * Float.BYTES; // segment offsets are in BYTES, not elements
            FloatVector va = FloatVector.fromMemorySegment(SPECIES, a, offset, ORDER);
            FloatVector vb = FloatVector.fromMemorySegment(SPECIES, b, offset, ORDER);
            va.add(vb).intoMemorySegment(c, offset, ORDER);
        }
        for (; i < n; i++) {
            c.setAtIndex(JAVA_FLOAT, i, a.getAtIndex(JAVA_FLOAT, i) + b.getAtIndex(JAVA_FLOAT, i));
        }
    }

    /** Dot product on off-heap memory, the same idea as lab 1's DotProduct. */
    public static float dot(MemorySegment a, MemorySegment b, long n) {
        long lanes = SPECIES.length();
        long upper = n - (n % lanes);
        FloatVector acc = FloatVector.zero(SPECIES);
        long i = 0;
        for (; i < upper; i += lanes) {
            long offset = i * Float.BYTES;
            acc = FloatVector.fromMemorySegment(SPECIES, a, offset, ORDER)
                    .fma(FloatVector.fromMemorySegment(SPECIES, b, offset, ORDER), acc);
        }
        float sum = acc.reduceLanes(jdk.incubator.vector.VectorOperators.ADD);
        for (; i < n; i++) sum += a.getAtIndex(JAVA_FLOAT, i) * b.getAtIndex(JAVA_FLOAT, i);
        return sum;
    }

    void main() {
        long n = 10_000_003;
        RandomGenerator rnd = RandomGenerator.getDefault();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment a = arena.allocate(JAVA_FLOAT, n);
            MemorySegment b = arena.allocate(JAVA_FLOAT, n);
            MemorySegment c = arena.allocate(JAVA_FLOAT, n);
            for (long i = 0; i < n; i++) {
                a.setAtIndex(JAVA_FLOAT, i, rnd.nextFloat());
                b.setAtIndex(JAVA_FLOAT, i, rnd.nextFloat());
            }

            add(a, b, c, n);
            long bad = 0;
            for (long i = 0; i < n; i++) {
                if (c.getAtIndex(JAVA_FLOAT, i) != a.getAtIndex(JAVA_FLOAT, i) + b.getAtIndex(JAVA_FLOAT, i)) bad++;
            }
            IO.println("off-heap vector add over %,d floats: %d mismatches".formatted(n, bad));

            Bench.measure("off-heap vector add", 20, 20, () -> add(a, b, c, n));
            Bench.measure("off-heap vector dot", 20, 20, () -> Bench.sink = dot(a, b, n));
            IO.println("dot(a, b) = " + dot(a, b, n));
        }
    }
}
