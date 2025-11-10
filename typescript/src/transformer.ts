import type {
  TransformerAPI,
  TokenizerLoadResult,
  TransformerLoadResult,
  EncodeResult,
  DecodeResult,
  GenerateResult,
  NextTokenResult,
} from "./types.js";

/**
 * Helper to wait for a global function to be available
 */
async function waitForGlobal(name: string, timeoutMs = 5000): Promise<void> {
  const t0 = performance.now();
  for (;;) {
    if ((globalThis as any)[name]) return;
    if (performance.now() - t0 > timeoutMs) {
      throw new Error(`Timeout waiting for ${name}`);
    }
    await new Promise((r) => setTimeout(r, 10));
  }
}

/**
 * Create transformer API wrapper around WASM functions
 */
export async function createTransformerAPI(): Promise<TransformerAPI> {
  // Wait for transformer functions to be available
  await Promise.all([
    waitForGlobal("LoadTokenizerFromBytes"),
    waitForGlobal("LoadTransformerFromBytes"),
    waitForGlobal("EncodeText"),
    waitForGlobal("DecodeTokens"),
    waitForGlobal("GenerateNextToken"),
    waitForGlobal("GenerateText"),
  ]);

  const g = globalThis as any;

  return {
    async loadTokenizer(
      tokenizerData: Uint8Array
    ): Promise<TokenizerLoadResult> {
      return new Promise((resolve, reject) => {
        try {
          const resultStr = g.LoadTokenizerFromBytes(tokenizerData);

          // If it's already an object, return it directly
          if (typeof resultStr === "object") {
            resolve(resultStr as TokenizerLoadResult);
            return;
          }

          const result = JSON.parse(resultStr);
          resolve(result);
        } catch (error) {
          reject(error);
        }
      });
    },

    async loadModel(
      configData: Uint8Array,
      weightsData: Uint8Array
    ): Promise<TransformerLoadResult> {
      return new Promise((resolve, reject) => {
        try {
          const resultStr = g.LoadTransformerFromBytes(configData, weightsData);
          const result = JSON.parse(resultStr);
          resolve(result);
        } catch (error) {
          reject(error);
        }
      });
    },

    async encode(
      text: string,
      addSpecialTokens: boolean = true
    ): Promise<EncodeResult> {
      return new Promise((resolve, reject) => {
        try {
          const resultStr = g.EncodeText(text, addSpecialTokens);
          const result = JSON.parse(resultStr);
          resolve(result);
        } catch (error) {
          reject(error);
        }
      });
    },

    async decode(
      tokenIds: number[],
      skipSpecialTokens: boolean = true
    ): Promise<DecodeResult> {
      return new Promise((resolve, reject) => {
        try {
          const resultStr = g.DecodeTokens(tokenIds, skipSpecialTokens);
          const result = JSON.parse(resultStr);
          resolve(result);
        } catch (error) {
          reject(error);
        }
      });
    },

    async generate(
      prompt: string,
      maxTokens: number = 50,
      temperature: number = 0.7
    ): Promise<GenerateResult> {
      return new Promise((resolve, reject) => {
        try {
          const resultStr = g.GenerateText(prompt, maxTokens, temperature);
          const result = JSON.parse(resultStr);
          resolve(result);
        } catch (error) {
          reject(error);
        }
      });
    },

    async *generateStream(
      prompt: string,
      maxTokens: number = 50,
      temperature: number = 0.7
    ): AsyncGenerator<string, void, unknown> {
      // Encode the prompt
      const encodeResultStr = g.EncodeText(prompt, true);
      const encodeResult: EncodeResult = JSON.parse(encodeResultStr);
      if (!encodeResult.success || !encodeResult.ids) {
        throw new Error(encodeResult.error || "Failed to encode prompt");
      }

      const tokens = [...encodeResult.ids];

      // Generate tokens one at a time
      for (let i = 0; i < maxTokens; i++) {
        const resultStr = g.GenerateNextToken(tokens, temperature);
        const result: NextTokenResult = JSON.parse(resultStr);

        if (!result.success) {
          throw new Error(result.error || "Failed to generate token");
        }

        if (result.token === undefined) {
          break;
        }

        tokens.push(result.token);

        // Decode just this token
        const decodeResultStr = g.DecodeTokens([result.token], true);
        const decodeResult: DecodeResult = JSON.parse(decodeResultStr);
        if (decodeResult.success && decodeResult.text) {
          yield decodeResult.text;
        }

        // Check for end of sequence
        if (result.is_eos) {
          break;
        }
      }
    },
  };
}
