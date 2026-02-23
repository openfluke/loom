package io.loom.example;

import io.loom.LoomException;
import io.loom.LoomNetwork;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * XOR training example using the Loom Java SDK.
 *
 * <p>Run:
 * <pre>
 *   mvn package -q
 *   java -jar target/loom-java-0.1.0.jar
 * </pre>
 * Requires {@code libloom.dll} / {@code libloom.so} on the library path.
 */
public class XorTrain {

    public static void main(String[] args) throws LoomException {
        System.out.println("🔷 Loom Java – XOR training example\n");

        // Generate layers: 11 layers of 2->2, 1 final layer of 2->1
        java.util.List<Map<String, Object>> layers = new java.util.ArrayList<>();
        for (int i = 0; i < 11; i++) {
            layers.add(Map.of("type", "dense", "activation", "relu", "input_size", 2, "output_size", 2));
        }
        layers.add(Map.of("type", "dense", "activation", "sigmoid", "input_size", 2, "output_size", 1));

        // Build a small 2-input network
        try (LoomNetwork net = LoomNetwork.create(new java.util.HashMap<>(Map.of(
                "grid_rows", 2,
                "grid_cols", 2,
                "layers_per_cell", 3,
                "input_size", 2,
                "layers", layers)))) {

            // XOR dataset
            List<float[]> inputs = List.of(
                    new float[]{0, 0},
                    new float[]{0, 1},
                    new float[]{1, 0},
                    new float[]{1, 1}
            );
            List<float[]> targets = List.of(
                    new float[]{0},
                    new float[]{1},
                    new float[]{1},
                    new float[]{0}
            );

            int epochs = 1000;
            float lr = 0.01f;

            for (int epoch = 0; epoch < epochs; epoch++) {
                for (int i = 0; i < inputs.size(); i++) {
                    net.forward(inputs.get(i));
                    float[] grad = {targets.get(i)[0] - 0.5f};
                    net.backward(grad);
                    net.updateWeights(lr);
                }
                if ((epoch + 1) % 200 == 0) {
                    System.out.printf("Epoch %d/%d%n", epoch + 1, epochs);
                }
            }

            // Test
            System.out.println("\nResults after " + epochs + " epochs:");
            for (int i = 0; i < inputs.size(); i++) {
                float[] out = net.forward(inputs.get(i));
                System.out.printf("  %s → %.4f (expected %.0f)%n",
                        Arrays.toString(inputs.get(i)),
                        out.length > 0 ? out[0] : -1,
                        targets.get(i)[0]);
            }

            // Save / load round-trip
            String json = net.save("xor");
            System.out.printf("%nModel saved (%d bytes)%n", json.length());
        }

        System.out.println("\n✅ Done.");
    }
}
