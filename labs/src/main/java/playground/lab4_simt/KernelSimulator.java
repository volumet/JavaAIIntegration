package playground.lab4_simt;

import java.util.stream.IntStream;

/**
 * Lab 4: a CPU simulator of the GPU (SIMT) programming model that HAT exposes.
 *
 * <p>Babylon/HAT is not part of JDK 25 (see the README for how to try the real one). The
 * <em>programming model</em> is the important part, though, and you can learn it on a CPU:
 * <ul>
 *   <li>You write a <b>kernel</b>: the work for ONE thread, identified by {@code kc.x}.</li>
 *   <li>The runtime launches {@code globalSize} logical threads in <b>groups</b> (CUDA: blocks).</li>
 *   <li>The global size gets rounded up to a whole number of groups, so a kernel must bounds-check
 *       {@code kc.x < kc.maxX}.</li>
 * </ul>
 *
 * <p>Mapping between HAT and CUDA (from the article):
 * <pre>
 *   HAT                       CUDA
 *   kc.x                      blockIdx.x * blockDim.x + threadIdx.x
 *   kc.maxX                   n  (problem size)
 *   kc.group.x / kc.group.maxX blockIdx.x / gridDim.x
 *   cc.dispatchKernel(n, k)   kernel&lt;&lt;&lt;blocks, threadsPerBlock&gt;&gt;&gt;(...)
 * </pre>
 */
public final class KernelSimulator {

    private KernelSimulator() {}

    /** Mirrors HAT's KernelContext (as a record): global id, global size, group id, group size. */
    public record KernelContext(int x, int maxX, int groupX, int groupSize) {}

    @FunctionalInterface
    public interface Kernel {
        void run(KernelContext kc);
    }

    /**
     * Launches {@code ceil(globalSize / groupSize)} groups of {@code groupSize} threads.
     * Groups run in parallel on the CPU's cores. Threads inside a group run one after another,
     * because a CPU core has no warp hardware. Real GPUs run them in lock-step.
     */
    public static void dispatch(int globalSize, int groupSize, Kernel kernel) {
        int groups = (globalSize + groupSize - 1) / groupSize;
        IntStream.range(0, groups).parallel().forEach(g -> {
            for (int t = 0; t < groupSize; t++) {
                kernel.run(new KernelContext(g * groupSize + t, globalSize, g, groupSize));
            }
        });
    }
}
