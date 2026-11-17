package playground.lab1_vector;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;

import java.util.List;

/**
 * Lab 1.0: what does SIMD look like on <em>your</em> CPU?
 *
 * <p>A {@link VectorSpecies} = element type + shape (bit width). {@code SPECIES_PREFERRED} is the
 * widest shape the CPU handles well (AVX2 gives 256 bits, AVX-512 gives 512, ARM NEON gives 128).
 *
 * <p>Note the Java 25 style: an instance {@code void main()} and {@code IO.println} (JEP 512).
 */
public class SpeciesInfo {

    void main() {
        VectorShape shape = VectorShape.preferredShape();
        IO.println("Preferred vector shape: " + shape + " = " + shape.vectorBitSize() + " bits");
        IO.println("Lanes per vector operation for each element type:");

        List<VectorSpecies<?>> all = List.of(
                ByteVector.SPECIES_PREFERRED, ShortVector.SPECIES_PREFERRED,
                IntVector.SPECIES_PREFERRED, LongVector.SPECIES_PREFERRED,
                FloatVector.SPECIES_PREFERRED, DoubleVector.SPECIES_PREFERRED);
        for (VectorSpecies<?> s : all) {
            IO.println("  %-7s %2d lanes".formatted(s.elementType().getSimpleName(), s.length()));
        }

        // One "instruction", many data: a fixed 128-bit vector of 4 floats.
        FloatVector v = FloatVector.fromArray(FloatVector.SPECIES_128, new float[] {1, 2, 3, 4}, 0);
        IO.println();
        IO.println("v           = " + v);
        IO.println("v * 2       = " + v.mul(2f));
        IO.println("v + v       = " + v.add(v));
        IO.println("sum(lanes)  = " + v.reduceLanes(VectorOperators.ADD));
        IO.println("v > 2 mask  = " + v.compare(VectorOperators.GT, 2f));
    }
}
