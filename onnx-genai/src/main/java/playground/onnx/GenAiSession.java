package playground.onnx;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.function.Consumer;

import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;
import static playground.onnx.foreign.ort_genai_c_h.*;

/**
 * A small, safe Java wrapper over the ONNX Runtime GenAI C API ({@code ort_genai_c.h}).
 *
 * <p>Everything named {@code Oga...} comes from the bindings that jextract generates during the
 * build ({@code target/generated-sources/jextract}). Open {@code ort_genai_c_h.java} next to
 * {@code ort_genai_c.h}: every C function became a static Java method with the same name.
 *
 * <p>Three C conventions show up again and again, and this class handles each in one place:
 * <ol>
 *   <li><b>Errors.</b> Most functions return {@code OgaResult*}. NULL means success. Anything
 *       else is an error object, which we turn into an exception and free.
 *   <li><b>Out-parameters.</b> {@code OgaCreateModel(path, OgaModel** out)} writes a pointer
 *       into memory we provide. We pass {@link #out} (8 bytes) and read the pointer back.
 *   <li><b>Ownership.</b> Every {@code OgaCreateX} needs a matching {@code OgaDestroyX}. Java's
 *       GC knows nothing about native objects, so {@link #close()} frees them in reverse order.
 * </ol>
 *
 * <p>Not thread-safe: the confined arena ties it to the thread that created it.
 */
public final class GenAiSession implements AutoCloseable {

    public record Stats(long promptTokens, int generatedTokens, long nanos, long contextTokens, boolean contextFull) {
        public double tokensPerSecond() {
            return generatedTokens / (nanos / 1e9);
        }
    }

    private final Arena arena = Arena.ofConfined();
    /** Reused "T** out" slot: big enough for one pointer. */
    private final MemorySegment out = arena.allocate(C_POINTER);
    /**
     * Reused "size_t* out_count" slot. Deliberately JAVA_LONG (8 bytes): size_t is 64-bit on
     * every 64-bit OS, while C_LONG is only 4 bytes on Windows (C's LLP64 model). Using
     * C_LONG here would let the native code write past the end of the segment.
     */
    private final MemorySegment count = arena.allocate(JAVA_LONG);

    private final int maxLength;
    private final String systemPrompt;
    private final MemorySegment model;
    private final MemorySegment tokenizer;

    // Per-conversation state: the generator holds the KV cache, i.e. the "memory" of the chat.
    private MemorySegment params = MemorySegment.NULL;
    private MemorySegment generator = MemorySegment.NULL;
    private MemorySegment stream = MemorySegment.NULL;
    private boolean firstTurn;

    public GenAiSession(Path modelDir, int maxLength, String systemPrompt) {
        this.maxLength = maxLength;
        this.systemPrompt = systemPrompt;
        try {
            model = create(OgaCreateModel(arena.allocateFrom(modelDir.toString()), out));
            tokenizer = create(OgaCreateTokenizer(model, out));
            newConversation();
        } catch (RuntimeException e) {
            close();
            throw e;
        }
    }

    /** Throws away the conversation (KV cache) and starts a fresh one. The model stays loaded. */
    public void newConversation() {
        destroyConversation();
        params = create(OgaCreateGeneratorParams(model, out));
        try (Arena tmp = Arena.ofConfined()) {
            check(OgaGeneratorParamsSetSearchNumber(params, tmp.allocateFrom("max_length"), maxLength));
        }
        generator = create(OgaCreateGenerator(model, params, out));
        stream = create(OgaCreateTokenizerStream(tokenizer, out));
        firstTurn = true;
    }

    /** Sends one user message and streams the reply, token by token, to {@code onText}. */
    public Stats ask(String userMessage, Consumer<String> onText) {
        String prompt = formatTurn(userMessage);
        long promptTokens;

        // 1. Text -> token ids, appended to the generator (which already holds earlier turns).
        MemorySegment sequences = create(OgaCreateSequences(out));
        try (Arena tmp = Arena.ofConfined()) {
            check(OgaTokenizerEncode(tokenizer, tmp.allocateFrom(prompt), sequences));
            promptTokens = OgaSequencesGetSequenceCount(sequences, 0);
            check(OgaGenerator_AppendTokenSequences(generator, sequences));
        } finally {
            OgaDestroySequences(sequences);
        }

        // 2. The decode loop: one forward pass through the transformer per generated token.
        long t0 = System.nanoTime();
        int generated = 0;
        while (!OgaGenerator_IsDone(generator)) {
            check(OgaGenerator_GenerateNextToken(generator));
            check(OgaGenerator_GetNextTokens(generator, out, count)); // out = const int32_t*, count = 1 here
            int token = out.get(C_POINTER, 0).reinterpret(JAVA_INT.byteSize()).get(JAVA_INT, 0);

            // 3. Token id -> text. The stream buffers partial UTF-8 / sub-word pieces for us.
            //    The returned char* belongs to the stream, so we must NOT free it.
            check(OgaTokenizerStreamDecode(stream, token, out));
            onText.accept(cString(out.get(C_POINTER, 0)));
            generated++;
        }
        long contextTokens = OgaGenerator_TokenCount(generator);
        return new Stats(promptTokens, generated, System.nanoTime() - t0, contextTokens, contextTokens >= maxLength);
    }

    /**
     * Builds the prompt text for one turn with the model's own chat template (from its
     * tokenizer_config.json), so this works for Phi-3, Llama, Qwen, ... without hard-coding
     * markers like {@code <|user|>}. Only the NEW turn is formatted, because the generator
     * already contains everything said before.
     */
    String formatTurn(String userMessage) {
        StringBuilder json = new StringBuilder("[");
        if (firstTurn && systemPrompt != null && !systemPrompt.isBlank()) {
            json.append(message("system", systemPrompt)).append(',');
        }
        json.append(message("user", userMessage)).append(']');
        firstTurn = false;

        try (Arena tmp = Arena.ofConfined()) {
            check(OgaTokenizerApplyChatTemplate(tokenizer, MemorySegment.NULL, tmp.allocateFrom(json.toString()),
                    MemorySegment.NULL, true, out));
            MemorySegment text = out.get(C_POINTER, 0);
            try {
                return cString(text);
            } finally {
                OgaDestroyString(text); // this string is ours: the API documents that we must free it
            }
        }
    }

    // ---------------------------------------------------------------- C-convention helpers

    /** Turns a non-NULL {@code OgaResult*} into an exception (and frees it). */
    private static void check(MemorySegment result) {
        if (!result.equals(MemorySegment.NULL)) {
            String message = cString(OgaResultGetError(result));
            OgaDestroyResult(result);
            throw new GenAiException(message);
        }
    }

    /** check(), then return the pointer the call wrote into {@link #out}. */
    private MemorySegment create(MemorySegment result) {
        check(result);
        return out.get(C_POINTER, 0);
    }

    /**
     * A {@code char*} from C arrives as a zero-length segment: Java knows the address but not
     * the size. {@code reinterpret} says "trust me, read until the NUL terminator". It is a
     * <em>restricted</em> method, hence {@code --enable-native-access}.
     */
    private static String cString(MemorySegment p) {
        return p.equals(MemorySegment.NULL) ? "" : p.reinterpret(Long.MAX_VALUE).getString(0);
    }

    private static String message(String role, String content) {
        return "{\"role\":\"" + role + "\",\"content\":" + jsonString(content) + "}";
    }

    static String jsonString(String s) {
        StringBuilder sb = new StringBuilder("\"");
        for (char c : s.toCharArray()) {
            switch (c) {
                case '"' -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                default -> {
                    if (c < 0x20) sb.append("\\u%04x".formatted((int) c));
                    else sb.append(c);
                }
            }
        }
        return sb.append('"').toString();
    }

    private void destroyConversation() {
        if (!stream.equals(MemorySegment.NULL)) OgaDestroyTokenizerStream(stream);
        if (!generator.equals(MemorySegment.NULL)) OgaDestroyGenerator(generator);
        if (!params.equals(MemorySegment.NULL)) OgaDestroyGeneratorParams(params);
        stream = generator = params = MemorySegment.NULL;
    }

    @Override
    public void close() {
        destroyConversation();
        // Fields may be unset if the constructor failed half-way.
        if (tokenizer != null && !tokenizer.equals(MemorySegment.NULL)) OgaDestroyTokenizer(tokenizer);
        if (model != null && !model.equals(MemorySegment.NULL)) OgaDestroyModel(model);
        arena.close();
    }

    public static final class GenAiException extends RuntimeException {
        GenAiException(String message) {
            super(message);
        }
    }
}
