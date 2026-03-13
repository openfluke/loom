# Loom – Java Bindings

Java bindings for the [Loom](https://github.com/openfluke/loom) neural-network inference engine, using **JNA (Java Native Access)** to call the native shared library directly — no JNI boilerplate required.

---

## Requirements

| Tool | Version |
|------|---------|
| JDK | ≥ 11 (21 LTS recommended) |
| Maven | ≥ 3.9 |
| libloom | built from `cabi/build_windows.sh` (or equivalent) |

---

## Build

```sh
cd wrapper/java
mvn package -q
```

This produces a fat JAR at `target/loom-java-0.1.0.jar`.

---

## Quick Start

```java
import io.loom.LoomNetwork;
import java.util.Map;

try (LoomNetwork net = LoomNetwork.create(Map.of(
        "grid_rows", 2, "grid_cols", 2,
        "layers_per_cell", 3, "input_size", 4))) {

    float[] out = net.forward(new float[]{1f, 0f, 0.5f, 0.3f});
    System.out.println(Arrays.toString(out));
    String json = net.save("my_model");
}
```

---

## API

### `LoomNetwork` (implements `AutoCloseable`)

| Method | Description |
|--------|-------------|
| `LoomNetwork.create(config)` | Build a network from a `Map<String,Object>` |
| `LoomNetwork.load(json)` | Restore a saved model |
| `forward(float[])` | Run a forward pass → `float[]` |
| `backward(float[])` | Backward pass with gradient vector |
| `updateWeights(float)` | SGD weight update |
| `applyAdamW(lr, b1, b2, wd)` | AdamW update |
| `trainStandard(inputs, targets, cfg)` | Batch training |
| `save(modelId)` | Serialise to JSON string |
| `evaluate(inputs, expected)` | Compute accuracy metrics |
| `close()` / try-with-resources | Release native resources |

### `LoomLibrary`

Raw JNA interface with every C export. Use this if you need access to
StepState, TweenState, AdaptationTracker, schedulers, or grafting APIs
directly.

---

## Running the example

```sh
# Build libloom first (from repo root):
# bash cabi/build_windows.sh   (WSL / Git Bash)
# Add libloom.dll to PATH or -Djna.library.path=<dir>

mvn package -q
java -jar target/loom-java-0.1.0.jar

# Or with explicit library path:
java -Djna.library.path=../../cabi/build/windows_x86_64 \
     -jar target/loom-java-0.1.0.jar
```

---

## Structure

```
java/
├── pom.xml
└── src/main/java/io/loom/
    ├── LoomLibrary.java      # JNA interface (all C exports)
    ├── LoomNetwork.java      # High-level API with AutoCloseable
    ├── LoomException.java    # Checked exception
    └── example/
        └── XorTrain.java     # Runnable XOR example
```
