package playground.lab4_simt;

import playground.lab4_simt.KernelSimulator.KernelContext;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Lab 4.2: kernels that need care.
 *
 * <ol>
 *   <li>Vector add: the article's CUDA {@code addArrays} kernel written in Java. Every thread
 *       writes its own element, so the kernel is "embarrassingly parallel" and has no races.</li>
 *   <li>Sum reduction: every thread wants to write the SAME place. The naive version races and
 *       gives a wrong total. Two fixes: an atomic add, or a per-group partial sum followed by a
 *       final pass. The second is how GPU reductions (and our Vector API reduceLanes) work.</li>
 * </ol>
 */
public class VectorAddKernel {

    // CUDA:
    //   __global__ void addArrays(int *a, int *b, int *c, int n) {
    //       int i = blockIdx.x * blockDim.x + threadIdx.x;
    //       if (i < n) c[i] = a[i] + b[i];
    //   }
    static void addArrays(KernelContext kc, int[] a, int[] b, int[] c) {
        if (kc.x() < kc.maxX()) {
            c[kc.x()] = a[kc.x()] + b[kc.x()];
        }
    }

    static long[] racySum = new long[1];

    void main() {
        int n = 1_000_000;
        int[] a = new int[n];
        int[] b = new int[n];
        int[] c = new int[n];
        Arrays.setAll(a, i -> i);
        Arrays.setAll(b, i -> 2 * i);

        KernelSimulator.dispatch(n, 256, kc -> addArrays(kc, a, b, c));
        boolean ok = true;
        for (int i = 0; i < n; i++) ok &= c[i] == 3 * i;
        IO.println("vector add correct: " + ok);

        long expected = (long) n * (n - 1) / 2;

        // 1. Racy: lost updates when two groups do read-modify-write at the same time
        KernelSimulator.dispatch(n, 256, kc -> {
            if (kc.x() < kc.maxX()) racySum[0] += a[kc.x()];
        });
        IO.println("racy sum   = %,d (expected %,d)%s".formatted(racySum[0], expected,
                racySum[0] == expected ? "  <- got lucky, run again" : "  <- WRONG, lost updates"));

        // 2. Atomic: correct, but every thread contends on one memory location
        AtomicLong atomic = new AtomicLong();
        KernelSimulator.dispatch(n, 256, kc -> {
            if (kc.x() < kc.maxX()) atomic.addAndGet(a[kc.x()]);
        });
        IO.println("atomic sum = %,d".formatted(atomic.get()));

        // 3. Two-level reduction: each group writes its own slot, then one small final pass
        int groupSize = 256;
        long[] partial = new long[(n + groupSize - 1) / groupSize];
        KernelSimulator.dispatch(n, groupSize, kc -> {
            if (kc.x() < kc.maxX()) partial[kc.groupX()] += a[kc.x()]; // only this group touches this slot
        });
        IO.println("tree sum   = %,d".formatted(Arrays.stream(partial).sum()));
    }
}
