# JavaAIIntegration

A hands-on playground for learning how Java runs AI workloads fast, following
[*Advancing AI by Accelerating Java on Parallel Architectures*](https://inside.java/2024/10/23/java-and-ai/)
(inside.java, Oct 2024).

That article is about the **engine room** of AI in Java: how the JDK uses SIMD units, native memory,
and GPUs. It does not cover calling LLM APIs. Labs 1–4 are small runnable programs that each end in
something AI-shaped: embedding search, matrix multiplication, or a neural-network forward pass.
Lab 5 follows a [Devoxx 2026 talk](https://www.youtube.com/watch?v=8dgnng27kMo) and runs a real LLM
locally through FFM and jextract.

## Setup

1. **Install JDK 25 or newer.** The code compiles with `release 25`, so it runs on any newer JDK too
   (tested on 25.0.2 and 27.0.1). Either works for every lab, lab 5 included, because jextract 25
   generates code for JDK 25. JDK 25 is the long-term-support release. JDK 27 is supported only until
   March 2027.
   - IntelliJ: *File > Project Structure > SDK > Add SDK > Download JDK*
   - `winget install EclipseAdoptium.Temurin.25.JDK`
2. **No system-wide `JAVA_HOME` change needed.** Other projects on this machine can stay on older JDKs.
   - IntelliJ: set *Project Structure > Project > SDK* to the 25+ JDK, and in *Settings > Build Tools >
     Maven > Runner* set *JRE* to "Use Project JDK".
   - `run.ps1` and `onnx-genai/chat.ps1` use `JAVA_HOME` when it is 25+. Otherwise they pick the
     newest 25+ JDK in `~/.jdks` (where IntelliJ downloads them) and leave `JAVA_HOME` alone.
   - `mvnd` and `mvnw` always use `JAVA_HOME`. For IntelliJ's terminal in this project only, set it in
     *Settings > Tools > Terminal > Environment variables*: `JAVA_HOME=C:\Users\<you>\.jdks\openjdk-27.0.1.0.3`.
     In any other shell, set it for that session: `$env:JAVA_HOME = "..."`.
3. Check it: `./run.ps1 lab1_vector/SpeciesInfo`

There are four ways to run a lab:

| How | Command |
|---|---|
| Source launcher (no build) | `./run.ps1 lab1_vector/DotProduct` (Git Bash: `./run.sh ...`) |
| [mvnd](https://github.com/apache/maven-mvnd) (Maven daemon, fastest repeated builds) | `mvnd -q -pl labs compile exec:exec -Dmain=playground.lab1_vector.DotProduct` |
| Maven wrapper (no install needed) | `./mvnw -q -pl labs compile exec:exec -Dmain=playground.lab1_vector.DotProduct` |
| IntelliJ | Open `pom.xml` as a project. In *Run > Edit Configurations > Edit configuration templates > Application*, add VM options `--add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED`. Lab 5 needs `-Dort.lib=...` and `-Dgenai.lib=...` too (see `chat.ps1`) |

Run the correctness tests with `mvnd test` or `./mvnw test`.

**mvnd notes**
- The daemon runs on whatever JDK `JAVA_HOME` points to. If you switch JDKs, mvnd starts a separate
  daemon for the new one. `mvnd --status` lists the running daemons, and `mvnd --stop` stops them all to free memory.
- `.mvn/maven.config` sets `-Dmvnd.rawStreams=true`. Without it, mvnd routes the lab's output through
  Maven's logger, and `-q` would hide it. Plain Maven ignores this property.
- mvnd 1.0.6 bundles Maven 3.9.16, and the wrapper is pinned to the same version.

> The `WARNING: Using incubator modules: jdk.incubator.vector` line is expected. The Vector API is
> still incubating in JDK 25 and 27; it is waiting for Project Valhalla's value classes. The FFM API is final.

The code also uses Java 25 features: instance `void main()` and `IO.println` (JEP 512), records, and
text blocks.

## Learning path

| # | Lab | Article section | What you'll see |
|---|---|---|---|
| 1.0 | `lab1_vector/SpeciesInfo` | Vector API | Your CPU's SIMD width, lanes, masks |
| 1.1 | `lab1_vector/ArrayAddition` | Vector API | The article's example: scalar vs. vector vs. masked. Why it's *not* faster (auto-vectorization, memory bandwidth) |
| 1.2 | `lab1_vector/DotProduct` | Vector API | Reductions + FMA: a real speed-up. Toy semantic search over 50k embeddings |
| 2.1 | `lab2_ffm/OffHeapBasics` | FFM: off-heap memory | `Arena`, `MemorySegment`, lifetimes, safety checks |
| 2.2 | `lab2_ffm/StructLayouts` | FFM: memory layouts | The article's `struct Point`, `VarHandle`s, padding |
| 2.3 | `lab2_ffm/NativeCalls` | FFM: (beyond the article) | Calling C `strlen`, and C `qsort` calling back into Java |
| 2.4 | `lab2_ffm/VectorOnSegments` | Vector API + FFM | SIMD directly on native memory, as pure-Java LLM engines do |
| 3.1 | `lab3_tinynn/MatMul` | (applying it) | Naive → cache-friendly → SIMD → multicore: from ~3 to ~300 GFLOP/s |
| 3.2 | `lab3_tinynn/TinyMLP` | (applying it) | Dense layers, ReLU, softmax; XOR by hand; scalar vs. SIMD inference |
| 4.1 | `lab4_simt/Squares` | Babylon / HAT | The article's HAT `squares` kernel on a CPU simulator of the SIMT model |
| 4.2 | `lab4_simt/VectorAddKernel` | GPU programming (SIMT) | The article's CUDA `addArrays` in Java; why reductions need care on GPUs |
| 5 | `onnx-genai/` → `playground.onnx.Chat` | (Devoxx talk, demo 1) | jextract bindings for ONNX Runtime GenAI; chat with Phi-3-mini locally. Needs manual downloads, see [Lab 5](#lab-5-chat-with-a-real-model-onnx-genai) |

Measured on the author's AVX2 machine (8 float lanes): dot product **3.3×**, MLP inference **~6×**,
and matmul **~90×** from naive to parallel SIMD. Your numbers will differ. Treat `Bench` as a
rough stopwatch. [JMH](https://github.com/openjdk/jmh) is the tool for real benchmarks.

## Exercises

Do these in order. Write a test in `LabsTest` for each one.

**Lab 1: Vector API**
1. Run `ArrayAddition` with `-XX:-UseSuperWord` (this disables auto-vectorization) and compare the numbers again.
2. Change `DotProduct.dotVector` to use **two** accumulators (unroll by 2). Does it get faster? Why? (Hint: FMA latency.)
3. Write `l2Distance(a, b)`, the Euclidean distance that some vector DBs use instead of cosine similarity.
4. Run with `FloatVector.SPECIES_128` instead of `SPECIES_PREFERRED`. How does lane count affect speed?

**Lab 2: FFM**
1. Memory-map a file of floats: `FileChannel.open(...).map(READ_WRITE, 0, size, arena)` returns a
   `MemorySegment`. Write 1M floats, reopen the file, and compute their dot product with `VectorOnSegments.dot`.
   This is exactly how LLM weights get loaded.
2. Call C's `memset` or `abs` via a downcall.
3. Allocate a segment bigger than 2 GB (if you have the RAM). You can't do that with a `float[]`.

**Lab 3: Tiny NN**
1. Vectorize `softmax` using `lanewise(VectorOperators.EXP)`.
2. Add a `forwardBatch(float[][] xs)` that turns the batch into a matrix multiply, and reuse `MatMul.parallelVector`.
3. Try cache **tiling/blocking** in `MatMul` (process 64×64 blocks). Measure n = 1024 and 2048.
4. Stretch: train XOR with backprop instead of hand-picked weights.

**Lab 4: SIMT**
1. Write a kernel for matrix multiply in which each thread computes one `C[i][j]` (use `x = i*n + j`).
2. Write a kernel for ReLU, then for one dense layer. Compare with lab 3.
3. Read the real HAT [`squares` example](https://github.com/openjdk/babylon/tree/code-reflection/hat/examples/squares)
   and map each line to `lab4_simt/Squares`.

## Lab 5: chat with a real model (`onnx-genai/`)

This module follows the first demo in [*Java and the Next Generation of Programming Models*](https://www.youtube.com/watch?v=8dgnng27kMo)
(Devoxx Belgium 2026, 10:30–18:00). It runs **Phi-3-mini** locally through
[ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai)'s C API, called from Java with FFM.
The Java bindings are generated by **jextract**. No JNI, no Python.

**How the build works**
1. The enforcer checks that the native downloads exist, and fails with the expected layout if not.
2. **jextract** turns `ort_genai_c.h` into Java under `target/generated-sources/jextract`.
   Its ~70 harmless warnings go to `target/jextract.log`.
3. Those generated files compile together with the module's own code.

The module is only part of the build when `tools/` exists, so a clone without the native downloads still builds.

**Native downloads** (git-ignored, downloaded by hand):
```
tools/jextract-25/                                    https://jdk.java.net/jextract/
tools/onnxruntime-genai/onnxruntime-genai-0.20.0-win-x64/  https://github.com/microsoft/onnxruntime-genai/releases
tools/onnxruntime/onnxruntime-win-x64-1.32.0/         https://github.com/microsoft/onnxruntime/releases
models/phi3-repo/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/
    git clone --filter=blob:none --no-checkout https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx models/phi3-repo
    git -C models/phi3-repo sparse-checkout set cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4
    git -C models/phi3-repo checkout main
```
Different versions or locations: pass `-Djextract.home=... -Dgenai.home=... -Dort.home=... -Dmodel.dir=...`
to Maven, and update the paths at the top of `onnx-genai/chat.ps1`, which accepts only `-Model`.

**Run it**
```powershell
mvnd -pl onnx-genai compile                                   # jextract + compile
./onnx-genai/chat.ps1                                         # interactive: /reset, /exit
./onnx-genai/chat.ps1 -Prompt "Explain a KV cache briefly"    # one-shot
mvnd -q -pl onnx-genai exec:exec "-Dprompt=Hello!"           # one-shot through mvnd
```
On an AVX2 laptop CPU it runs at about 17–18 tokens/s, and the model loads in about 4 s.

**What to read in the code** (`GenAiSession.java`):
- **Error results.** Every `OgaResult*` that isn't NULL is an error. It becomes an exception and gets freed.
- **Out-parameters.** For `T** out`, we pass a pointer-sized segment and read the pointer back.
- **`size_t*`.** It's allocated as `JAVA_LONG`, not `C_LONG`: a C `long` is 4 bytes on Windows
  but `size_t` is 8. The upstream jextract sample uses `C_LONG` here, which overflows on Windows.
- **Strings.** `reinterpret(...)` reads a `char*` of unknown length, and `OgaDestroyString` frees
  only the strings the API says you own.
- **Chat format.** `OgaTokenizerApplyChatTemplate` uses the model's own template, so prompts
  aren't hard-coded to Phi-3's `<|user|>` markers.
- **Conversation memory.** The generator keeps the KV cache, so each turn only appends its new
  tokens. `/reset` destroys the generator and creates a fresh one.

`Chat.java` loads `onnxruntime.dll` **before** `onnxruntime-genai.dll`, and explains why (Windows DLL search order).
It also calls `OgaShutdown` through `makeInvoker()`, because the header declares it as `void OgaShutdown()`.
In C, empty parentheses mean "unspecified arguments", so jextract treats it like a variadic function.

**Exercises**
1. Add `--temperature` and `--top-p` options (`OgaGeneratorParamsSetSearchNumber`, plus `do_sample` via `...SetSearchBool`).
2. Print each token's id next to its text to see how words split into tokens.
3. Implement `/undo` with `OgaGenerator_RewindTo` to drop the last exchange.
4. Try another model exported for ONNX Runtime GenAI (e.g. a small Qwen or Llama) by changing only `-Model`.
5. Rewrite `GenAiSession` so native objects are freed by `segment.reinterpret(arena, OgaDestroyX)`
   (the upstream sample's approach). What order does the arena release them in, and does it matter?

## Going further

- **Real Babylon/HAT.** It isn't in any JDK release yet (not even 27). Build the `code-reflection` branch of
  [openjdk/babylon](https://github.com/openjdk/babylon) and follow the README under `hat/`.
  You'll need a GPU backend (CUDA or OpenCL). The same repo's `cr-examples/onnx` is the talk's
  second demo (Llama written in Java with `@Reflect`, from 31:00): a natural lab 6.
- **TornadoVM** ([beehive-lab/TornadoVM](https://github.com/beehive-lab/TornadoVM)) runs Java on
  GPUs today, and its `@Parallel` loops mirror lab 4.
- **The same techniques in real AI projects**: [Jlama](https://github.com/tjake/Jlama) and
  [llama3.java](https://github.com/mukel/llama3.java) run LLMs in pure Java using the Vector API
  plus memory-mapped `MemorySegment`s (labs 1–3, at full scale).
- **The application layer** (calling LLMs, RAG, agents):
  [LangChain4j](https://github.com/langchain4j/langchain4j) and
  [Spring AI](https://spring.io/projects/spring-ai). These are a good next module for this repo.

## Layout

```
pom.xml                    parent: shared versions; adds onnx-genai only when tools/ exists
labs/src/main/java/playground/
  common/Bench.java        tiny timing helper
  lab1_vector/             Vector API (SIMD)
  lab2_ffm/                Foreign Function & Memory API
  lab3_tinynn/             matmul + MLP built on labs 1-2
  lab4_simt/               GPU/SIMT programming model, simulated on the CPU
labs/src/test/java/playground/LabsTest.java   every fast version is checked against the plain version
onnx-genai/                lab 5: jextract bindings + local LLM chat (GenAiSession, Chat, chat.ps1)
tools/, models/            native downloads and model weights (git-ignored)
docker/                    Redis 8 vector store for RAG (Part B): docker-compose.redis-vector.yml, port 6380
run.ps1 / run.sh           source-launcher runners for labs 1-4 (no build)
find-jdk.ps1               picks a 25+ JDK for the scripts without touching JAVA_HOME
mvnw / mvnw.cmd            Maven wrapper (downloads Maven 3.9.16 on first use)
.mvn/maven.config          shared Maven/mvnd options
```
