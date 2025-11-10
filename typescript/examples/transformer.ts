/**
 * Example: Transformer inference with LOOM WASM
 *
 * This example demonstrates how to:
 * 1. Load a tokenizer and transformer model
 * 2. Encode/decode text
 * 3. Generate text with streaming
 */

import { initLoom, createTransformerAPI } from "@openfluke/welvet";

async function main() {
  console.log("=== LOOM TypeScript Transformer Example ===\n");

  // Initialize WASM
  console.log("1. Initializing WASM...");
  await initLoom();
  console.log("   ✓ WASM initialized\n");

  // Create transformer API
  console.log("2. Creating transformer API...");
  const transformer = await createTransformerAPI();
  console.log("   ✓ Transformer API ready\n");

  // Load tokenizer
  console.log("3. Loading tokenizer...");
  const tokenizerResponse = await fetch(
    "models/SmolLM2-135M-Instruct/tokenizer.json"
  );
  const tokenizerData = new Uint8Array(await tokenizerResponse.arrayBuffer());
  const tokResult = await transformer.loadTokenizer(tokenizerData);

  if (!tokResult.success) {
    console.error(`   ✗ Failed: ${tokResult.error}`);
    return;
  }
  console.log(`   ✓ Loaded tokenizer with ${tokResult.vocab_size} tokens\n`);

  // Load model
  console.log("4. Loading transformer model...");
  const configResponse = await fetch(
    "models/SmolLM2-135M-Instruct/config.json"
  );
  const weightsResponse = await fetch(
    "models/SmolLM2-135M-Instruct/model.safetensors"
  );

  const configData = new Uint8Array(await configResponse.arrayBuffer());
  const weightsData = new Uint8Array(await weightsResponse.arrayBuffer());

  const modelResult = await transformer.loadModel(configData, weightsData);

  if (!modelResult.success) {
    console.error(`   ✗ Failed: ${modelResult.error}`);
    return;
  }
  console.log(`   ✓ Loaded model:`);
  console.log(`     - Layers: ${modelResult.num_layers}`);
  console.log(`     - Hidden size: ${modelResult.hidden_size}`);
  console.log(`     - Vocab size: ${modelResult.vocab_size}\n`);

  // Test encoding
  console.log("5. Testing text encoding...");
  const testText = "Once upon a time";
  const encodeResult = await transformer.encode(testText, true);

  if (!encodeResult.success) {
    console.error(`   ✗ Failed: ${encodeResult.error}`);
    return;
  }
  console.log(`   Input: "${testText}"`);
  console.log(`   Token IDs: [${encodeResult.ids?.join(", ")}]\n`);

  // Test decoding
  console.log("6. Testing token decoding...");
  const decodeResult = await transformer.decode(encodeResult.ids!, true);

  if (!decodeResult.success) {
    console.error(`   ✗ Failed: ${decodeResult.error}`);
    return;
  }
  console.log(`   Decoded: "${decodeResult.text}"\n`);

  // Test streaming generation
  console.log("7. Testing streaming generation...");
  const prompt = "The capital of France is";
  console.log(`   Prompt: "${prompt}"`);
  process.stdout.write("   Generated: ");

  let tokenCount = 0;
  try {
    for await (const token of transformer.generateStream(prompt, 30, 0.7)) {
      process.stdout.write(token);
      tokenCount++;
    }
    console.log(`\n   Generated ${tokenCount} tokens\n`);
  } catch (error) {
    console.error(`\n   ✗ Error: ${error}`);
    return;
  }

  console.log("✓ All tests completed!");
}

main().catch(console.error);
