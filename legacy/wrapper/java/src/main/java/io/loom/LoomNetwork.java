package io.loom;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.jna.Pointer;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * High-level Java API for the Loom neural-network inference engine.
 *
 * <p>Wraps the global-network C ABI exported by {@code libloom}
 * via {@link LoomLibrary}. All JSON marshalling is handled internally;
 * callers work with plain Java arrays and maps.
 *
 * <pre>{@code
 * try (LoomNetwork net = LoomNetwork.create(Map.of(
 *         "grid_rows", 2, "grid_cols", 2,
 *         "layers_per_cell", 3, "input_size", 4))) {
 *     float[] out = net.forward(new float[]{1, 0, 0.5f, 0.3f});
 *     System.out.println(Arrays.toString(out));
 * }
 * }</pre>
 */
public final class LoomNetwork implements AutoCloseable {

    private static final LoomLibrary LIB = LoomLibrary.INSTANCE;
    private static final ObjectMapper JSON = new ObjectMapper();

    private LoomNetwork() {}

    // -------------------------------------------------------------------------
    // Factory methods
    // -------------------------------------------------------------------------

    /**
     * Create a network from a config map.
     *
     * <p>Typical keys: {@code grid_rows}, {@code grid_cols},
     * {@code layers_per_cell}, {@code input_size}, {@code batch_size},
     * {@code use_gpu}.
     */
    public static LoomNetwork create(Map<String, Object> config) throws LoomException {
        String json = toJson(config);
        Pointer ptr = LIB.CreateLoomNetwork(json);
        String result = ptrToString(ptr);
        checkError(result);
        return new LoomNetwork();
    }

    /**
     * Restore a previously saved model from its JSON representation.
     */
    public static LoomNetwork load(String modelJson, String modelId) throws LoomException {
        Pointer ptr = LIB.LoomLoadModel(modelJson, modelId);
        String result = ptrToString(ptr);
        checkError(result);
        return new LoomNetwork();
    }

    public static LoomNetwork load(String modelJson) throws LoomException {
        return load(modelJson, "model");
    }

    // -------------------------------------------------------------------------
    // Inference
    // -------------------------------------------------------------------------

    /**
     * Run a forward pass.
     *
     * @param inputs float input vector
     * @return output vector
     */
    public float[] forward(float[] inputs) throws LoomException {
        Pointer ptr = LIB.LoomForward(inputs, inputs.length);
        String json = ptrToString(ptr);
        try {
            JsonNode node = JSON.readTree(json);
            if (node.isObject() && node.has("error")) {
                throw new LoomException(node.get("error").asText());
            }
            float[] out = new float[node.size()];
            for (int i = 0; i < out.length; i++) {
                out[i] = (float) node.get(i).asDouble();
            }
            return out;
        } catch (JsonProcessingException e) {
            throw new LoomException("Failed to parse forward result: " + e.getMessage());
        }
    }

    /**
     * Run a backward pass with the given gradient vector.
     */
    public void backward(float[] gradients) throws LoomException {
        Pointer ptr = LIB.LoomBackward(gradients, gradients.length);
        checkError(ptrToString(ptr));
    }

    /** SGD weight update. */
    public void updateWeights(float learningRate) {
        LIB.LoomUpdateWeights(learningRate);
    }

    /** Apply AdamW gradients. */
    public void applyAdamW(float lr, float beta1, float beta2, float weightDecay) {
        LIB.LoomApplyGradientsAdamW(lr, beta1, beta2, weightDecay);
    }

    /** Apply RMSprop gradients. */
    public void applyRMSprop(float lr, float alpha, float epsilon, float momentum) {
        LIB.LoomApplyGradientsRMSprop(lr, alpha, epsilon, momentum);
    }

    // -------------------------------------------------------------------------
    // Training
    // -------------------------------------------------------------------------

    /**
     * Train with 2-D inputs / targets.
     *
     * @param inputs  List of input float arrays
     * @param targets List of target float arrays
     * @param config  Training config map (e.g. {@code epochs}, {@code learning_rate})
     */
    public Map<String, Object> trainStandard(
            List<float[]> inputs,
            List<float[]> targets,
            Map<String, Object> config) throws LoomException {
        String iJson = toJson(inputs);
        String tJson = toJson(targets);
        String cJson = toJson(config);
        Pointer ptr = LIB.LoomTrainStandard(iJson, tJson, cJson);
        return parseMap(ptrToString(ptr));
    }

    // -------------------------------------------------------------------------
    // Persistence
    // -------------------------------------------------------------------------

    /**
     * Serialise the current network to a JSON string.
     */
    public String save(String modelId) throws LoomException {
        Pointer ptr = LIB.LoomSaveModel(modelId);
        return ptrToString(ptr);
    }

    public String save() throws LoomException {
        return save("model");
    }

    // -------------------------------------------------------------------------
    // Info / Evaluation
    // -------------------------------------------------------------------------

    /** Return basic network metadata. */
    public Map<String, Object> info() throws LoomException {
        return parseMap(ptrToString(LIB.LoomGetNetworkInfo()));
    }

    /** Evaluate accuracy. */
    public Map<String, Object> evaluate(
            List<float[]> inputs, double[] expectedOutputs) throws LoomException {
        String iJson = toJson(inputs);
        String eJson = toJson(expectedOutputs);
        return parseMap(ptrToString(LIB.LoomEvaluateNetwork(iJson, eJson)));
    }

    /** Sync GPU state (no-op on CPU-only builds). */
    public void syncGPU() {
        LIB.LoomSyncGPU();
    }

    // -------------------------------------------------------------------------
    // AutoCloseable
    // -------------------------------------------------------------------------

    @Override
    public void close() {
        LIB.FreeLoomNetwork();
    }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    /**
     * Convert a native {@code char*} pointer to a Java string and free it.
     */
    public static String ptrToString(Pointer ptr) {
        if (ptr == null) return "";
        try {
            return ptr.getString(0);
        } finally {
            LIB.FreeLoomString(ptr);
        }
    }

    private static void checkError(String json) throws LoomException {
        try {
            JsonNode node = JSON.readTree(json);
            if (node.isObject() && node.has("error")) {
                throw new LoomException(node.get("error").asText());
            }
        } catch (JsonProcessingException e) {
            // not JSON — treat as raw string, probably fine
        }
    }

    private static String toJson(Object obj) throws LoomException {
        try {
            return JSON.writeValueAsString(obj);
        } catch (JsonProcessingException e) {
            throw new LoomException("JSON serialisation failed: " + e.getMessage());
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> parseMap(String json) throws LoomException {
        try {
            Map<String, Object> m = JSON.readValue(json, Map.class);
            if (m.containsKey("error")) throw new LoomException((String) m.get("error"));
            return m;
        } catch (JsonProcessingException e) {
            throw new LoomException("Failed to parse response: " + e.getMessage());
        }
    }
}
