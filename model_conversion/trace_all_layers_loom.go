package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/openfluke/loom/nn"
)

func main() {
	fmt.Println("=================================================================")
	fmt.Println(" LOOM All Layers Trace - Comparing with PyTorch")
	fmt.Println("=================================================================")

	// Load model
	modelPath := "/home/samuel/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
	fmt.Printf("\nLoading model...\n")

	network, err := nn.LoadTransformerFromSafetensors(modelPath)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}

	// Load embeddings separately
	weightsPath := filepath.Join(modelPath, "model.safetensors")
	tensors, err := nn.LoadSafetensors(weightsPath)
	if err != nil {
		fmt.Printf("Error loading weights: %v\n", err)
		os.Exit(1)
	}

	embeddings := tensors["model.embed_tokens.weight"]
	hiddenSize := network.InputSize
	vocabSize := len(embeddings) / hiddenSize

	fmt.Printf("Model loaded: hidden=%d, vocab=%d, layers=%d\n", hiddenSize, vocabSize, len(network.Layers))

	// Input tokens: "Once upon a time" = [12522, 5193, 264, 882]
	inputTokens := []int{12522, 5193, 264, 882}
	fmt.Printf("Input tokens: %v\n", inputTokens)
	fmt.Printf("Sequence length: %d\n", len(inputTokens))

	seqLen := len(inputTokens)
	lastIdx := (seqLen - 1) * hiddenSize

	// Step 1: Embedding
	fmt.Println("\n============================================================")
	fmt.Println("STEP 1: Token Embedding")
	fmt.Println("============================================================")
	hidden := make([]float32, seqLen*hiddenSize)
	for i, tokenID := range inputTokens {
		copy(hidden[i*hiddenSize:(i+1)*hiddenSize],
			embeddings[tokenID*hiddenSize:(tokenID+1)*hiddenSize])
	}

	fmt.Printf("Shape: [%d, %d]\n", seqLen, hiddenSize)
	fmt.Printf("Last token embedding[842]: %.10f\n", hidden[lastIdx+842])

	pytorchEmb, _ := loadNpy("/tmp/pytorch_trace/embedding.npy")
	if pytorchEmb != nil {
		diff := hidden[lastIdx+842] - pytorchEmb[lastIdx+842]
		if math.Abs(float64(diff)) < 1e-6 {
			fmt.Printf("  ✓ Match with PyTorch (diff: %.2e)\n", diff)
		} else {
			fmt.Printf("  ✗ MISMATCH with PyTorch (diff: %.2e)\n", diff)
		}
	}

	// Process all 24 transformer layers
	numTransformerBlocks := 24
	for block := 0; block < numTransformerBlocks; block++ {
		fmt.Printf("\n============================================================\n")
		fmt.Printf("LAYER %d\n", block)
		fmt.Printf("============================================================\n")

		// Each transformer block has 4 layers: [InputNorm, Attention, PostAttnNorm, SwiGLU]
		layerBase := block * 4
		inputNorm := &network.Layers[layerBase+0]
		attn := &network.Layers[layerBase+1]
		postAttnNorm := &network.Layers[layerBase+2]
		swiGLU := &network.Layers[layerBase+3]

		fmt.Printf("\nInput to layer %d:\n", block)
		fmt.Printf("  Last token[842]: %.10f\n", hidden[lastIdx+842])

		// Sublayer 1: Input LayerNorm
		fmt.Println("\n  Sublayer 1: Input LayerNorm")
		normed := applyRMSNorm(hidden, inputNorm.Gamma, inputNorm.Epsilon, hiddenSize)
		fmt.Printf("    After norm, last token[842]: %.10f\n", normed[lastIdx+842])

		pytorchNorm, _ := loadNpy(fmt.Sprintf("/tmp/pytorch_trace/layer_%d_input_norm.npy", block))
		if pytorchNorm != nil {
			diff := normed[lastIdx+842] - pytorchNorm[lastIdx+842]
			if math.Abs(float64(diff)) < 1e-5 {
				fmt.Printf("    ✓ Match (diff: %.2e)\n", diff)
			} else {
				fmt.Printf("    ✗ MISMATCH (diff: %.2e)\n", diff)
			}
		}

		// Sublayer 2: Self Attention
		fmt.Println("\n  Sublayer 2: Self Attention")
		attnOutput, _ := nn.MultiHeadAttentionForwardCPU(normed, attn, 1)
		fmt.Printf("    Attention output, last token[842]: %.10f\n", attnOutput[lastIdx+842])

		pytorchAttn, _ := loadNpy(fmt.Sprintf("/tmp/pytorch_trace/layer_%d_attn_output.npy", block))
		if pytorchAttn != nil {
			diff := attnOutput[lastIdx+842] - pytorchAttn[lastIdx+842]
			if math.Abs(float64(diff)) < 1e-4 {
				fmt.Printf("    ✓ Match (diff: %.2e)\n", diff)
			} else {
				fmt.Printf("    ✗ MISMATCH (diff: %.2e)\n", diff)
			}
		}

		// Add residual 1
		for i := range hidden {
			hidden[i] += attnOutput[i]
		}
		fmt.Printf("    After residual, last token[842]: %.10f\n", hidden[lastIdx+842])

		pytorchResidual, _ := loadNpy(fmt.Sprintf("/tmp/pytorch_trace/layer_%d_after_attn_residual.npy", block))
		if pytorchResidual != nil {
			diff := hidden[lastIdx+842] - pytorchResidual[lastIdx+842]
			if math.Abs(float64(diff)) < 1e-4 {
				fmt.Printf("    ✓ Match (diff: %.2e)\n", diff)
			} else {
				fmt.Printf("    ✗ MISMATCH (diff: %.2e)\n", diff)
			}
		}

		// Sublayer 3: Post-attention LayerNorm
		fmt.Println("\n  Sublayer 3: Post-attention LayerNorm")
		normed2 := applyRMSNorm(hidden, postAttnNorm.Gamma, postAttnNorm.Epsilon, hiddenSize)
		fmt.Printf("    After norm, last token[842]: %.10f\n", normed2[lastIdx+842])

		pytorchNorm2, _ := loadNpy(fmt.Sprintf("/tmp/pytorch_trace/layer_%d_post_attn_norm.npy", block))
		if pytorchNorm2 != nil {
			diff := normed2[lastIdx+842] - pytorchNorm2[lastIdx+842]
			if math.Abs(float64(diff)) < 1e-5 {
				fmt.Printf("    ✓ Match (diff: %.2e)\n", diff)
			} else {
				fmt.Printf("    ✗ MISMATCH (diff: %.2e)\n", diff)
			}
		}

		// Sublayer 4: MLP (SwiGLU)
		fmt.Println("\n  Sublayer 4: MLP")
		mlpOutput := applySwiGLU(normed2, swiGLU, hiddenSize, seqLen)
		fmt.Printf("    MLP output, last token[842]: %.10f\n", mlpOutput[lastIdx+842])

		pytorchMLP, _ := loadNpy(fmt.Sprintf("/tmp/pytorch_trace/layer_%d_mlp_output.npy", block))
		if pytorchMLP != nil {
			diff := mlpOutput[lastIdx+842] - pytorchMLP[lastIdx+842]
			if math.Abs(float64(diff)) < 1e-3 {
				fmt.Printf("    ✓ Match (diff: %.2e)\n", diff)
			} else {
				fmt.Printf("    ✗ MISMATCH (diff: %.2e)\n", diff)
			}
		}

		// Add residual 2
		for i := range hidden {
			hidden[i] += mlpOutput[i]
		}
		fmt.Printf("    After residual, last token[842]: %.10f\n", hidden[lastIdx+842])

		pytorchLayerOut, _ := loadNpy(fmt.Sprintf("/tmp/pytorch_trace/layer_%d_output.npy", block))
		if pytorchLayerOut != nil {
			diff := hidden[lastIdx+842] - pytorchLayerOut[lastIdx+842]
			if math.Abs(float64(diff)) < 1e-3 {
				fmt.Printf("    ✓ Match (diff: %.2e)\n", diff)
			} else {
				fmt.Printf("    ✗ MISMATCH (diff: %.2e)\n", diff)
			}
		}

		fmt.Printf("\n  Layer %d complete ✓\n", block)
	}

	// Final norm
	fmt.Println("\n============================================================")
	fmt.Println("Final Layer Normalization")
	fmt.Println("============================================================")
	finalNorm := tensors["model.norm.weight"]
	finalNormed := applyRMSNorm(hidden, finalNorm, 1e-6, hiddenSize)
	fmt.Printf("After final norm, last token[842]: %.10f\n", finalNormed[lastIdx+842])

	pytorchFinalNorm, _ := loadNpy("/tmp/pytorch_trace/final_norm.npy")
	if pytorchFinalNorm != nil {
		diff := finalNormed[lastIdx+842] - pytorchFinalNorm[lastIdx+842]
		if math.Abs(float64(diff)) < 1e-3 {
			fmt.Printf("  ✓ Match (diff: %.2e)\n", diff)
		} else {
			fmt.Printf("  ✗ MISMATCH (diff: %.2e)\n", diff)
		}
	}

	// LM head projection
	fmt.Println("\n============================================================")
	fmt.Println("LM Head Projection")
	fmt.Println("============================================================")

	// Check if lm_head exists, otherwise use tied weights (embeddings)
	lmHead, hasLMHead := tensors["lm_head.weight"]
	if !hasLMHead {
		fmt.Println("Using tied weights (embeddings as LM head)")
		lmHead = embeddings
	}

	lastToken := finalNormed[lastIdx : lastIdx+hiddenSize]

	logits := make([]float32, vocabSize)
	for v := 0; v < vocabSize; v++ {
		sum := float32(0)
		for d := 0; d < hiddenSize; d++ {
			sum += lastToken[d] * lmHead[v*hiddenSize+d]
		}
		logits[v] = sum
	}

	// Find top 5
	type TokenLogit struct {
		token int
		logit float32
	}
	top5 := make([]TokenLogit, 5)
	for i := 0; i < 5; i++ {
		maxIdx := 0
		maxVal := logits[0]
		for j := 1; j < vocabSize; j++ {
			if logits[j] > maxVal {
				maxVal = logits[j]
				maxIdx = j
			}
		}
		top5[i] = TokenLogit{maxIdx, maxVal}
		logits[maxIdx] = -1e9
	}

	fmt.Println("Top 5 predictions:")
	for i, tl := range top5 {
		fmt.Printf("  %d. Token %6d (logit: %9.4f)\n", i+1, tl.token, tl.logit)
	}

	fmt.Printf("\nPredicted next token: %d\n", top5[0].token)

	// Compare with PyTorch
	pytorchLogits, _ := loadNpy("/tmp/pytorch_trace/logits.npy")
	if pytorchLogits != nil {
		// PyTorch logits are shape [1, 4, 151936], we need last token: [0, -1, :]
		lastTokenLogits := pytorchLogits[3*vocabSize : 4*vocabSize]

		// Find PyTorch's top token
		pytorchMaxIdx := 0
		pytorchMaxVal := lastTokenLogits[0]
		for j := 1; j < vocabSize; j++ {
			if lastTokenLogits[j] > pytorchMaxVal {
				pytorchMaxVal = lastTokenLogits[j]
				pytorchMaxIdx = j
			}
		}
		fmt.Printf("PyTorch predicted: %d (logit: %.4f)\n", pytorchMaxIdx, pytorchMaxVal)

		if top5[0].token == pytorchMaxIdx {
			fmt.Println("  ✓ Same token predicted!")
		} else {
			fmt.Println("  ✗ Different token predicted")
		}
	}

	fmt.Println("\n=================================================================")
	fmt.Println(" ✅ All layers trace complete")
	fmt.Println("=================================================================")
}

func applyRMSNorm(input []float32, gamma []float32, epsilon float32, dModel int) []float32 {
	seqLen := len(input) / dModel
	output := make([]float32, len(input))

	for t := 0; t < seqLen; t++ {
		offset := t * dModel

		// Compute RMS (root mean square)
		sumSquares := float32(0)
		for i := 0; i < dModel; i++ {
			val := input[offset+i]
			sumSquares += val * val
		}
		rms := float32(math.Sqrt(float64(sumSquares/float32(dModel) + epsilon)))

		// Normalize and apply scale (gamma)
		for i := 0; i < dModel; i++ {
			normalized := input[offset+i] / rms
			gammaIdx := i
			if gammaIdx < len(gamma) {
				output[offset+i] = normalized * gamma[gammaIdx]
			} else {
				output[offset+i] = normalized
			}
		}
	}

	return output
}

func applySwiGLU(input []float32, swiGLU *nn.LayerConfig, dModel int, seqLen int) []float32 {
	intermediateDim := len(swiGLU.GateWeights) / dModel
	output := make([]float32, len(input))

	for t := 0; t < seqLen; t++ {
		// Gate projection with SiLU activation
		gate := make([]float32, intermediateDim)
		for i := 0; i < intermediateDim; i++ {
			sum := float32(0)
			for j := 0; j < dModel; j++ {
				sum += input[t*dModel+j] * swiGLU.GateWeights[j*intermediateDim+i]
			}
			// SiLU: x * sigmoid(x)
			gate[i] = sum / (1.0 + float32(math.Exp(-float64(sum))))
		}

		// Up projection (linear)
		up := make([]float32, intermediateDim)
		for i := 0; i < intermediateDim; i++ {
			sum := float32(0)
			for j := 0; j < dModel; j++ {
				sum += input[t*dModel+j] * swiGLU.UpWeights[j*intermediateDim+i]
			}
			up[i] = sum
		}

		// Element-wise multiply
		intermediate := make([]float32, intermediateDim)
		for i := 0; i < intermediateDim; i++ {
			intermediate[i] = gate[i] * up[i]
		}

		// Down projection
		for i := 0; i < dModel; i++ {
			sum := float32(0)
			for j := 0; j < intermediateDim; j++ {
				sum += intermediate[j] * swiGLU.DownWeights[j*dModel+i]
			}
			output[t*dModel+i] = sum
		}
	}

	return output
}

func loadNpy(filename string) ([]float32, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Skip NPY header
	header := make([]byte, 128)
	n, _ := file.Read(header)

	dataStart := 0
	for i := 0; i < n; i++ {
		if header[i] == '\n' {
			dataStart = i + 1
			break
		}
	}

	file.Seek(int64(dataStart), 0)

	// Read float32 data
	stat, _ := file.Stat()
	numFloats := (int(stat.Size()) - dataStart) / 4
	data := make([]float32, numFloats)

	err = binary.Read(file, binary.LittleEndian, &data)
	if err != nil {
		return nil, err
	}

	return data, nil
}
