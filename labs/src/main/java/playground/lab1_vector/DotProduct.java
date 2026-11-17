package playground.lab1_vector;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import playground.common.Bench;

import java.util.Comparator;
import java.util.random.RandomGenerator;
import java.util.stream.IntStream;

/**
 * Lab 1.2: the dot product, the workhorse of AI. It powers embedding similarity, every neuron, and
 * every attention score.
 *
 * <p>The scalar loop is a <em>reduction</em> ({@code s += a[i]*b[i]}). Java requires float
 * additions to happen in source order, so the JIT may not reorder them into parallel lanes and
 * will NOT auto-vectorize this loop. With the Vector API we choose that reordering ourselves: we
 * keep one partial sum per lane and use fused multiply-add, then combine the lanes at the end.
 * The result can differ from the scalar sum in the last few bits. That is fine, and expected.
 *
 * <p>The second half builds a toy semantic search: brute-force cosine similarity over 50k
 * random "embeddings". This is exactly what a vector database does with no index.
 */
public class DotProduct {

    static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    public static float dotScalar(float[] a, float[] b) {
        float sum = 0f;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    public static float dotVector(float[] a, float[] b) {
        FloatVector acc = FloatVector.zero(SPECIES);
        int i = 0;
        int upper = SPECIES.loopBound(a.length);
        for (; i < upper; i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, i);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, i);
            acc = va.fma(vb, acc); // acc += va * vb, one rounding step, one partial sum per lane
        }
        float sum = acc.reduceLanes(VectorOperators.ADD);
        for (; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    /** Scales v to unit length in place. Afterwards, the dot product of two vectors IS their cosine similarity. */
    public static void normalize(float[] v) {
        float norm = (float) Math.sqrt(dotVector(v, v));
        for (int i = 0; i < v.length; i++) v[i] /= norm;
    }

    void main() {
        RandomGenerator rnd = RandomGenerator.getDefault();

        // --- Part 1: raw dot product speed -------------------------------------------------
        int n = 1 << 20;
        float[] a = new float[n];
        float[] b = new float[n];
        for (int i = 0; i < n; i++) {
            a[i] = rnd.nextFloat() - 0.5f;
            b[i] = rnd.nextFloat() - 0.5f;
        }
        IO.println("dot of %,d floats: scalar = %.6f, vector = %.6f (differences in the last digits are expected)"
                .formatted(n, dotScalar(a, b), dotVector(a, b)));
        double s = Bench.measure("scalar dot", 50, 50, () -> Bench.sink = dotScalar(a, b));
        double v = Bench.measure("vector dot (fma)", 50, 50, () -> Bench.sink = dotVector(a, b));
        IO.println("  speed-up: %.1fx%n".formatted(s / v));

        // --- Part 2: toy semantic search ---------------------------------------------------
        int docs = 50_000;
        int dim = 384; // the size of a popular small sentence-embedding model's output
        float[][] index = new float[docs][dim];
        for (float[] e : index) {
            for (int i = 0; i < dim; i++) e[i] = (float) rnd.nextGaussian();
            normalize(e);
        }
        // The query is document #42 plus noise, so #42 should come out on top.
        float[] query = index[42].clone();
        for (int i = 0; i < dim; i++) query[i] += 0.02f *(float) rnd.nextGaussian();
        normalize(query);

        float[] scores = new float[docs];
        IO.println("Brute-force cosine search over %,d x %d embeddings:".formatted(docs, dim));
        Bench.measure("scalar search", 5, 10, () -> {
            for (int d = 0; d < docs; d++) scores[d] = dotScalar(index[d], query);
        });
        Bench.measure("vector search", 5, 10, () -> {
            for (int d = 0; d < docs; d++) scores[d] = dotVector(index[d], query);
        });

        IO.println("Top 3 matches:");
        IntStream.range(0, docs).boxed()
                .sorted(Comparator.comparingDouble(d -> -scores[d]))
                .limit(3)
                .forEach(d -> IO.println("  doc #%-6d cosine = %.4f".formatted(d, scores[d])));
    }
}
