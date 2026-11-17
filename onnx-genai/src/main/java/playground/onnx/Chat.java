package playground.onnx;

import java.nio.file.Files;
import java.nio.file.Path;

import static playground.onnx.foreign.ort_genai_c_h.OgaShutdown;

/**
 * Lab 5: chat with a local LLM (Phi-3-mini, ONNX) from plain Java via FFM. No JNI, no Python.
 *
 * <pre>
 *   Chat &lt;modelDir&gt; [--prompt=TEXT] [--max-length=N] [--system=TEXT]
 * </pre>
 * Without {@code --prompt} it starts an interactive chat. Commands: {@code /reset}, {@code /exit}.
 *
 * <p>Native libraries come from {@code -Dort.lib=...} and {@code -Dgenai.lib=...} (full paths),
 * or else from {@code java.library.path}.
 */
public class Chat {

    static final String DEFAULT_SYSTEM = "You are a helpful and accurate assistant. Answer concisely.";

    void main(String[] args) {
        if (args.length == 0) {
            IO.println("usage: Chat <modelDir> [--prompt=TEXT] [--max-length=N] [--system=TEXT]");
            System.exit(2);
        }
        Path modelDir = Path.of(args[0]);
        String prompt = option(args, "prompt", "");
        int maxLength = Integer.parseInt(option(args, "max-length", "2048"));
        String system = option(args, "system", DEFAULT_SYSTEM);

        if (!Files.isRegularFile(modelDir.resolve("genai_config.json"))) {
            IO.println("No genai_config.json in " + modelDir.toAbsolutePath()
                    + "\nPoint the first argument at the model folder (the one with the .onnx file).");
            System.exit(1);
        }

        loadNativeLibraries();

        IO.println("Loading " + modelDir.getFileName() + " ...");
        long t0 = System.nanoTime();
        try (GenAiSession session = new GenAiSession(modelDir, maxLength, system)) {
            IO.println("Loaded in %.1f s (context window: %d tokens)".formatted((System.nanoTime() - t0) / 1e9, maxLength));

            if (!prompt.isBlank()) {
                IO.println("\n> " + prompt);
                answer(session, prompt);
                return;
            }

            IO.println("Type a message. /reset starts a new conversation, /exit quits.");
            String line;
            while ((line = IO.readln("\n> ")) != null) {
                line = line.strip();
                if (line.isEmpty()) continue;
                if (line.equals("/exit")) break;
                if (line.equals("/reset")) {
                    session.newConversation();
                    IO.println("(new conversation)");
                    continue;
                }
                if (!answer(session, line)) {
                    session.newConversation();
                    IO.println("(context window full: started a new conversation)");
                }
            }
        } finally {
            // After every Oga object is destroyed (try-with-resources closed first).
            // The header declares `void OgaShutdown()`. In C, empty parentheses mean "unspecified
            // arguments", NOT "no arguments" (that would be `(void)`). So jextract treats it like
            // a variadic function: you build an invoker for the argument types, here none.
            OgaShutdown.makeInvoker().apply();
        }
    }

    /** Streams one reply. Returns false when the context window is full. */
    private static boolean answer(GenAiSession session, String userMessage) {
        GenAiSession.Stats s = session.ask(userMessage, text -> {
            System.out.print(text);
            System.out.flush();
        });
        IO.println("\n[%d prompt tokens, %d generated, %.1f tokens/s, context %d/%s]".formatted(
                s.promptTokens(), s.generatedTokens(), s.tokensPerSecond(), s.contextTokens(),
                s.contextFull() ? "FULL" : "ok"));
        return !s.contextFull();
    }

    /**
     * Loads onnxruntime FIRST, then onnxruntime-genai. On Windows, when a DLL needs another DLL,
     * the loader does not look in the first DLL's own folder: it looks in the exe's folder,
     * system folders and PATH. Once onnxruntime.dll is already loaded in the process, though,
     * Windows reuses it. That is why the order matters.
     *
     * <p>The jextract bindings find their symbols via {@code SymbolLookup.loaderLookup()}, which
     * sees libraries loaded with System.load/loadLibrary.
     */
    static void loadNativeLibraries() {
        load("ort.lib", "onnxruntime");
        load("genai.lib", "onnxruntime-genai");
    }

    private static void load(String property, String libraryName) {
        String path = System.getProperty(property, "");
        if (path.isBlank()) {
            System.loadLibrary(libraryName);
        } else {
            System.load(Path.of(path).toAbsolutePath().normalize().toString());
        }
    }

    private static String option(String[] args, String name, String defaultValue) {
        String prefix = "--" + name + "=";
        for (String a : args) {
            if (a.startsWith(prefix)) return a.substring(prefix.length());
        }
        return defaultValue;
    }
}
