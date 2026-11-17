package playground.lab2_ffm;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.Arrays;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * Lab 2.3: calling C from Java without JNI or C glue code.
 *
 * <ul>
 *   <li><b>downcall</b>: Java calls C {@code strlen}</li>
 *   <li><b>upcall</b>: C {@code qsort} calls back into a Java comparator</li>
 * </ul>
 *
 * <p>This mechanism is how Java binds to native AI runtimes (llama.cpp, ONNX Runtime, CUDA, BLAS).
 * For real libraries you don't write these descriptors by hand. The {@code jextract} tool
 * generates them from C header files.
 *
 * <p>Run with {@code --enable-native-access=ALL-UNNAMED} or you'll get a warning; the run scripts
 * already pass it.
 */
public class NativeCalls {

    static int compareInts(MemorySegment a, MemorySegment b) {
        return Integer.compare(a.get(JAVA_INT, 0), b.get(JAVA_INT, 0));
    }

    void main() throws Throwable {
        Linker linker = Linker.nativeLinker();
        SymbolLookup stdlib = linker.defaultLookup(); // the C standard library of this platform

        // size_t strlen(const char*);   (size_t is 64-bit on x64 / aarch64)
        MethodHandle strlen = linker.downcallHandle(
                stdlib.find("strlen").orElseThrow(),
                FunctionDescriptor.of(JAVA_LONG, ADDRESS));

        // void qsort(void* base, size_t count, size_t size, int (*cmp)(const void*, const void*));
        MethodHandle qsort = linker.downcallHandle(
                stdlib.find("qsort").orElseThrow(),
                FunctionDescriptor.ofVoid(ADDRESS, JAVA_LONG, JAVA_LONG, ADDRESS));

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment cString = arena.allocateFrom("Hello from Panama!"); // UTF-8, NUL-terminated
            long len = (long) strlen.invokeExact(cString);
            IO.println("strlen(\"Hello from Panama!\") = " + len);

            // Turn a static Java method into a C function pointer that lives as long as the arena.
            MethodHandle cmp = MethodHandles.lookup().findStatic(NativeCalls.class, "compareInts",
                    MethodType.methodType(int.class, MemorySegment.class, MemorySegment.class));
            FunctionDescriptor cmpDesc = FunctionDescriptor.of(JAVA_INT,
                    ADDRESS.withTargetLayout(JAVA_INT), ADDRESS.withTargetLayout(JAVA_INT));
            MemorySegment cmpPtr = linker.upcallStub(cmp, cmpDesc, arena);

            int[] data = {42, 7, 19, 3, 88, 1, 56};
            MemorySegment nativeArray = arena.allocateFrom(JAVA_INT, data);
            qsort.invokeExact(nativeArray, (long) data.length, JAVA_INT.byteSize(), cmpPtr);
            IO.println("qsort(" + Arrays.toString(data) + ") = " + Arrays.toString(nativeArray.toArray(JAVA_INT)));
        }
    }
}
