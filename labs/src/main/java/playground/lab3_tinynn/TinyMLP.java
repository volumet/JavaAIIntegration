package playground.lab3_tinynn;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import playground.common.Bench;

import java.util.Arrays;
import java.util.random.RandomGenerator;

/**
 * Lab 3.2: a neural network forward pass (inference), built from the pieces in lab 1.
 *
 * <p>A dense (fully-connected) layer computes {@code y = W.x + b}: one dot product per output
 * neuron. Then an activation function runs element-wise (ReLU, softmax).
 *
 * <p>Part 1 solves XOR with hand-picked weights, so you can check every number by hand.
 * Part 2 runs an MNIST-shaped network (784 -> 256 -> 10) with random weights and times scalar
 * against SIMD inference.
 */
public class TinyMLP {

    static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    /** Weights are out x in, row-major: the row for output neuron o starts at o * in. */
    public record Dense(int in, int out, float[] w, float[] b) {
        public Dense {
            if (w.length != in * out || b.length != out) throw new IllegalArgumentException("shape mismatch");
        }

        public static Dense random(int in, int out, RandomGenerator rnd) {
            float[] w = new float[in * out];
            float scale = (float) Math.sqrt(2.0 / in); // He initialization, which suits ReLU
            for (int i = 0; i < w.length; i++) w[i] = (float) rnd.nextGaussian() * scale;
            return new Dense(in, out, w, new float[out]);
        }

        public float[] forward(float[] x) {
            float[] y = new float[out];
            for (int o = 0; o < out; o++) y[o] = dotVector(w, o * in, x, in) + b[o];
            return y;
        }

        public float[] forwardScalar(float[] x) {
            float[] y = new float[out];
            for (int o = 0; o < out; o++) {
                float s = b[o];
                for (int i = 0; i < in; i++) s += w[o * in + i] * x[i];
                y[o] = s;
            }
            return y;
        }
    }

    /** Dot product of w[off .. off+len) and x[0 .. len). */
    static float dotVector(float[] w, int off, float[] x, int len) {
        FloatVector acc = FloatVector.zero(SPECIES);
        int i = 0;
        int upper = SPECIES.loopBound(len);
        for (; i < upper; i += SPECIES.length()) {
            acc = FloatVector.fromArray(SPECIES, w, off + i).fma(FloatVector.fromArray(SPECIES, x, i), acc);
        }
        float sum = acc.reduceLanes(VectorOperators.ADD);
        for (; i < len; i++) sum += w[off + i] * x[i];
        return sum;
    }

    /** ReLU(v) = max(v, 0), in place and vectorized, with a mask for the tail. */
    public static float[] relu(float[] v) {
        for (int i = 0; i < v.length; i += SPECIES.length()) {
            VectorMask<Float> m = SPECIES.indexInRange(i, v.length);
            FloatVector.fromArray(SPECIES, v, i, m).lanewise(VectorOperators.MAX, 0f).intoArray(v, i, m);
        }
        return v;
    }

    public static float[] reluScalar(float[] v) {
        for (int i = 0; i < v.length; i++) v[i] = Math.max(v[i], 0f);
        return v;
    }

    /** Turns raw scores (logits) into probabilities. Subtracting the max avoids exp() overflow. */
    public static float[] softmax(float[] v) {
        float max = Float.NEGATIVE_INFINITY;
        for (float f : v) max = Math.max(max, f);
        float sum = 0f;
        float[] p = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            p[i] = (float) Math.exp(v[i] - max);
            sum += p[i];
        }
        for (int i = 0; i < p.length; i++) p[i] /= sum;
        return p;
    }

    static int argmax(float[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) if (v[i] > v[best]) best = i;
        return best;
    }

    void main() {
        // --- Part 1: XOR, with weights you can verify by hand --------------------------------
        //   h1 = relu(x1 + x2)      h2 = relu(x1 + x2 - 1)      out = h1 - 2*h2
        Dense hidden = new Dense(2, 2, new float[] {1, 1, 1, 1}, new float[] {0, -1});
        Dense output = new Dense(2, 1, new float[] {1, -2}, new float[] {0});
        IO.println("XOR network:");
        for (float[] x : new float[][] {{0, 0}, {0, 1}, {1, 0}, {1, 1}}) {
            float[] h = relu(hidden.forward(x));
            float y = output.forward(h)[0];
            IO.println("  %s -> hidden %s -> %.0f".formatted(Arrays.toString(x), Arrays.toString(h), y));
        }

        // --- Part 2: MNIST-shaped network, scalar vs SIMD inference --------------------------
        RandomGenerator rnd = RandomGenerator.getDefault();
        Dense l1 = Dense.random(784, 256, rnd);
        Dense l2 = Dense.random(256, 10, rnd);
        int batch = 2_000;
        float[][] images = new float[batch][784];
        for (float[] img : images) for (int i = 0; i < img.length; i++) img[i] = rnd.nextFloat();

        IO.println("%nMLP 784 -> 256 -> 10, %,d random 'images':".formatted(batch));
        Bench.measure("scalar inference", 3, 5, () -> {
            int votes = 0;
            for (float[] img : images) votes += argmax(l2.forwardScalar(reluScalar(l1.forwardScalar(img))));
            Bench.sink = votes;
        });
        Bench.measure("Vector API inference", 3, 5, () -> {
            int votes = 0;
            for (float[] img : images) votes += argmax(l2.forward(relu(l1.forward(img))));
            Bench.sink = votes;
        });

        float[] probs = softmax(l2.forward(relu(l1.forward(images[0]))));
        IO.println("Image #0 class probabilities (the weights are random, so the answer means nothing; training is what makes it meaningful):");
        for (int c = 0; c < probs.length; c++) IO.println("  class %d: %5.1f%%".formatted(c, probs[c] * 100));
    }
}
