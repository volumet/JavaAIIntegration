package playground.lab2_ffm;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;

/**
 * Lab 2.1: off-heap memory with the Foreign Function &amp; Memory API (final since JDK 22).
 *
 * <p>Why AI cares: model weights are big (GBs), get memory-mapped from disk, and are shared with
 * native code such as BLAS, CUDA, or llama.cpp. Java arrays live on the GC heap, can be moved by
 * the GC, and are limited to 2^31 elements. A {@link MemorySegment} is a bounds-checked view of
 * memory with a 64-bit size and a lifetime controlled by an {@link Arena}.
 *
 * <ul>
 *   <li>{@code Arena.ofConfined()}: one owner thread, freed at close (try-with-resources)</li>
 *   <li>{@code Arena.ofShared()}: any thread may access it, freed at close</li>
 *   <li>{@code Arena.ofAuto()}: freed by the GC when unreachable</li>
 *   <li>{@code Arena.global()}: never freed</li>
 * </ul>
 */
public class OffHeapBasics {

    void main() throws InterruptedException {
        int n = 8;
        MemorySegment escaped;

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment a = arena.allocate(JAVA_FLOAT, n);
            MemorySegment b = arena.allocateFrom(JAVA_FLOAT, 10, 20, 30, 40, 50, 60, 70, 80);
            MemorySegment c = arena.allocate(JAVA_FLOAT, n);

            for (int i = 0; i < n; i++) {
                a.setAtIndex(JAVA_FLOAT, i, i);
            }
            for (int i = 0; i < n; i++) {
                c.setAtIndex(JAVA_FLOAT, i, a.getAtIndex(JAVA_FLOAT, i) + b.getAtIndex(JAVA_FLOAT, i));
            }
            IO.println("c = a + b = " + Arrays.toString(c.toArray(JAVA_FLOAT)));
            IO.println("c is %d bytes of native memory at address 0x%x".formatted(c.byteSize(), c.address()));

            // A slice is a zero-copy view of part of a segment.
            MemorySegment middle = c.asSlice(2 * JAVA_FLOAT.byteSize(), 3 * JAVA_FLOAT.byteSize());
            IO.println("slice [2..5) = " + Arrays.toString(middle.toArray(JAVA_FLOAT)));

            // Confined means confined: another thread may not touch this memory.
            Thread.ofVirtual().start(() -> {
                try {
                    c.getAtIndex(JAVA_FLOAT, 0);
                } catch (WrongThreadException e) {
                    IO.println("from another thread -> " + e.getClass().getSimpleName());
                }
            }).join();

            escaped = c;
        } // <- all three segments are freed here, deterministically, with no GC involved

        // Safety: unlike C, you can't use memory after it was freed...
        try {
            escaped.getAtIndex(JAVA_FLOAT, 0);
        } catch (IllegalStateException e) {
            IO.println("use after free  -> " + e.getClass().getSimpleName() + ": " + e.getMessage());
        }
        // ...and you can't read past the end.
        try (Arena arena = Arena.ofConfined()) {
            arena.allocate(JAVA_FLOAT, 4).getAtIndex(JAVA_FLOAT, 4);
        } catch (IndexOutOfBoundsException e) {
            IO.println("out of bounds   -> " + e.getClass().getSimpleName());
        }
    }
}
