package main

import (
	"fmt"
	"log"
	"runtime"
	"runtime/debug"

	"github.com/openfluke/loom/poly"
)

type inferenceConfig struct {
	useGPU            bool
	useTiling         bool
	tilingMode        string
	tileSize          int
	weightDType       poly.DType
	sequentialGPULoad bool
	numLayers         int
	hiddenSize        int
	isQwen            bool
	useBitNetCPU      bool
	useTernaryPTQCPU  bool
	rmsNormEps        float64
	fromEntity        bool
}

func normalizeInferenceConfig(cfg *inferenceConfig) {
	if cfg.useGPU && cfg.useTiling && cfg.hiddenSize >= 1536 {
		fmt.Printf("⚠️  Large model detected (hidden=%d). Tiled GPU path can destabilize logits here; forcing Standard Forward.\n", cfg.hiddenSize)
		cfg.useTiling = false
		cfg.tilingMode = "1"
		cfg.tileSize = 0
	}
	if cfg.useGPU && cfg.hiddenSize >= 1536 && cfg.weightDType == poly.DTypeInt4 {
		fmt.Printf("⚠️  Large model detected (hidden=%d). Q4 can degrade output quality; promoting weight precision to INT8.\n", cfg.hiddenSize)
		cfg.weightDType = poly.DTypeInt8
	}
	if cfg.useGPU && cfg.isQwen && cfg.weightDType == poly.DTypeInt4 {
		fmt.Println("⚠️  Qwen GPU + Q4 is experimental and may reduce output quality.")
	}
	if cfg.useGPU && cfg.isQwen && cfg.weightDType == poly.DTypeInt8 {
		fmt.Println("ℹ️  Qwen GPU + INT8 enabled.")
	}
	if cfg.useGPU && cfg.isQwen && cfg.useTiling {
		fmt.Println("⚠️  Qwen GPU tiled path is experimental and may reduce output quality.")
	}
}

func setupTransformerForInference(tr *poly.Transformer[float32], cfg inferenceConfig) bool {
	if tr == nil || tr.Network == nil {
		log.Fatal("setupTransformerForInference: nil transformer")
	}
	normalizeInferenceConfig(&cfg)

	tr.SetRMSNormEps(cfg.rmsNormEps)
	for i := range tr.Network.Layers {
		tr.Network.Layers[i].MaxSeqLen = maxSeqLen
	}
	if cfg.useTiling {
		tr.EnableTiling(cfg.tileSize)
	}

	useGPU := cfg.useGPU
	if useGPU {
		if cfg.sequentialGPULoad && !cfg.fromEntity {
			fmt.Printf("⏳ GPU init + block-wise weight upload (%d transformer blocks)...\n", cfg.numLayers)
		} else if cfg.sequentialGPULoad && cfg.fromEntity {
			fmt.Printf("⏳ GPU init + block-wise ENTITY upload (%d transformer blocks)...\n", cfg.numLayers)
		} else {
			fmt.Print("⏳ GPU Synchronization... ")
		}
		if err := tr.Network.InitWGPU(); err != nil {
			if cfg.sequentialGPULoad {
				log.Fatalf("❌ GPU init required for block-wise load: %v", err)
			}
			fmt.Printf("❌ Failed: %v\n", err)
			useGPU = false
		} else {
			applyGlitchTilingFlags(tr.Network, true, cfg.useTiling, cfg.tilingMode)
			if cfg.sequentialGPULoad {
				for li := 0; li < cfg.numLayers; li++ {
					base := li * 4
					for j := 0; j < 4; j++ {
						idx := base + j
						layer := &tr.Network.Layers[idx]
						if layer.Type == poly.LayerRMSNorm {
							layer.DType = poly.DTypeFloat32
						} else {
							layer.DType = cfg.weightDType
							if layer.WeightStore != nil && cfg.weightDType != poly.DTypeFloat32 && !cfg.fromEntity {
								if _, ok := layer.WeightStore.Versions[cfg.weightDType]; !ok {
									layer.WeightStore.Morph(cfg.weightDType)
								}
							}
						}
						if err := layer.SyncToGPU(); err != nil {
							log.Fatalf("❌ GPU sync block %d layer %d: %v", li, j, err)
						}
					}
					for j := 0; j < 4; j++ {
						(&tr.Network.Layers[base+j]).ReleaseInferenceHostWeights()
					}
					fmt.Printf("   ✓ Block %d/%d on GPU\n", li+1, cfg.numLayers)
				}
			} else {
				for i := range tr.Network.Layers {
					if tr.Network.Layers[i].Type == poly.LayerRMSNorm {
						tr.Network.Layers[i].DType = poly.DTypeFloat32
					} else {
						tr.Network.Layers[i].DType = cfg.weightDType
						if tr.Network.Layers[i].WeightStore != nil && cfg.weightDType != poly.DTypeFloat32 && !cfg.fromEntity {
							if _, ok := tr.Network.Layers[i].WeightStore.Versions[cfg.weightDType]; !ok {
								tr.Network.Layers[i].WeightStore.Morph(cfg.weightDType)
							}
						}
					}
					if err := (&tr.Network.Layers[i]).SyncToGPU(); err != nil {
						log.Fatalf("❌ GPU sync layer %d: %v", i, err)
					}
				}
			}
			if err := tr.SyncToGPU(); err != nil {
				log.Fatalf("❌ Embedding / LM head GPU sync: %v", err)
			}

			_, _ = tr.ForwardTokenIDsWGPU([]uint32{0}, nil, true, true)
			tr.Reset()
			tr.ReleaseInferenceHostWeights()
			runtime.GC()
			debug.FreeOSMemory()
			fmt.Println("✅ Success!")
		}
	}
	if !useGPU {
		applyGlitchTilingFlags(tr.Network, false, cfg.useTiling, cfg.tilingMode)
		if cfg.useBitNetCPU || cfg.useTernaryPTQCPU {
			tr.Network.UseExactDType = true
		}
		tr.SyncInferenceCPU()
	}
	return useGPU
}
