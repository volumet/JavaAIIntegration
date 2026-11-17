package playground.lab4_simt;

import playground.lab4_simt.KernelSimulator.KernelContext;

import java.util.Arrays;

/**
 * Lab 4.1: the article's HAT "squares" example, run on the CPU simulator.
 *
 * <p>The real HAT version (from the article and openjdk/babylon hat/examples/squares) looks like:
 * <pre>{@code
 * @CodeReflection public static int squareit(int v) { return v * v; }
 *
 * @CodeReflection public static void squareKernel(KernelContext kc, S32Array s32Array) {
 *     if (kc.x < kc.maxX) {
 *         int value = s32Array.array(kc.x);
 *         s32Array.array(kc.x, squareit(value));
 *     }
 * }
 *
 * @CodeReflection public static void square(ComputeContext cc, S32Array s32Array) {
 *     cc.dispatchKernel(s32Array.length(), kc -> squareKernel(kc, s32Array));
 * }
 * }</pre>
 * {@code @CodeReflection} makes javac keep a "code model" of these methods. At run time HAT
 * translates that model into CUDA C, OpenCL C, or SPIR-V and runs it on the GPU, and S32Array is an
 * FFM MemorySegment that the GPU can read directly. Here, a plain int[] and the simulator stand in.
 */
public class Squares {

    static int squareit(int v) {
        return v * v;
    }

    static void squareKernel(KernelContext kc, int[] array) {
        if (kc.x() < kc.maxX()) { // remove this check and watch the last group crash
            array[kc.x()] = squareit(array[kc.x()]);
        }
    }

    void main() {
        int[] array = new int[30]; // not a multiple of the group size, on purpose
        Arrays.setAll(array, i -> i);

        KernelSimulator.dispatch(array.length, 8, kc -> squareKernel(kc, array));
        IO.println("squares = " + Arrays.toString(array));

        // Watch the scheduling: which group handled which ids? (output order varies from run to run)
        IO.println("\nGroup 0..3 x 8 threads over 30 elements:");
        KernelSimulator.dispatch(30, 8, kc -> {
            if (kc.x() % 8 == 0) {
                IO.println("  group %d on %s starts at x=%d".formatted(kc.groupX(), Thread.currentThread().getName(), kc.x()));
            }
        });
    }
}
