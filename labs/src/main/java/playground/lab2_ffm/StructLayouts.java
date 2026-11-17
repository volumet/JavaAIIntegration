package playground.lab2_ffm;

import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemoryLayout.PathElement;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.invoke.VarHandle;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;

/**
 * Lab 2.2: describing C structs with {@link MemoryLayout}, the article's {@code struct Point}.
 *
 * <pre>{@code
 * struct Point { int x; int y; };
 * struct Embedding { int id; float vec[4]; };   // a tiny record you might pass to a native library
 * }</pre>
 *
 * <p>A layout gives you sizes, offsets, and {@link VarHandle}s that read and write fields by name,
 * with no hand-computed offsets.
 */
public class StructLayouts {

    static final StructLayout POINT = MemoryLayout.structLayout(
            JAVA_INT.withName("x"),
            JAVA_INT.withName("y")
    ).withName("Point");

    // Coordinates of an arrayElementVarHandle: (segment, long baseOffset, long index)
    static final VarHandle X = POINT.arrayElementVarHandle(PathElement.groupElement("x"));
    static final VarHandle Y = POINT.arrayElementVarHandle(PathElement.groupElement("y"));

    static final StructLayout EMBEDDING = MemoryLayout.structLayout(
            JAVA_INT.withName("id"),
            MemoryLayout.sequenceLayout(4, JAVA_FLOAT).withName("vec")
    ).withName("Embedding");

    // Coordinates: (segment, long baseOffset, long vecIndex)
    static final VarHandle VEC = EMBEDDING.varHandle(PathElement.groupElement("vec"), PathElement.sequenceElement());

    void main() {
        IO.println(POINT + "  size=" + POINT.byteSize()
                + "  offset(x)=" + POINT.byteOffset(PathElement.groupElement("x"))
                + "  offset(y)=" + POINT.byteOffset(PathElement.groupElement("y")));
        IO.println(EMBEDDING + "  size=" + EMBEDDING.byteSize());

        try (Arena arena = Arena.ofConfined()) {
            // An array of 5 Points: struct Point points[5];
            int n = 5;
            MemorySegment points = arena.allocate(POINT, n);
            for (long i = 0; i < n; i++) {
                X.set(points, 0L, i, (int) i);
                Y.set(points, 0L, i, (int) (i * i));
            }
            StringBuilder sb = new StringBuilder("points = ");
            for (long i = 0; i < n; i++) {
                sb.append("(%d,%d) ".formatted((int) X.get(points, 0L, i), (int) Y.get(points, 0L, i)));
            }
            IO.println(sb);

            MemorySegment emb = arena.allocate(EMBEDDING);
            emb.set(JAVA_INT, EMBEDDING.byteOffset(PathElement.groupElement("id")), 42);
            for (long i = 0; i < 4; i++) VEC.set(emb, 0L, i, 0.25f * i);
            IO.println("embedding id=" + emb.get(JAVA_INT, 0) + " vec[3]=" + (float) VEC.get(emb, 0L, 3L));
        }

        // Alignment: an int must start at a multiple of 4 bytes. The API refuses a layout that breaks
        // this rule. C compilers insert this padding silently. In Java you write it explicitly.
        try {
            MemoryLayout.structLayout(JAVA_BYTE.withName("tag"), JAVA_INT.withName("value"));
        } catch (IllegalArgumentException e) {
            IO.println("{byte, int} without padding -> " + e.getClass().getSimpleName());
        }
        StructLayout padded = MemoryLayout.structLayout(
                JAVA_BYTE.withName("tag"), MemoryLayout.paddingLayout(3), JAVA_INT.withName("value"));
        IO.println("{byte, pad(3), int} -> " + padded + " size=" + padded.byteSize());
    }
}
