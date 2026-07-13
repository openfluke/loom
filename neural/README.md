# neural

Thin MNIST façade over **`poly.NeuralFountain`**.

For arbitrary models / layers / datasets, use the poly API:

```go
factory := poly.DenseSpecialistFactory("my-net", sizes, nil)
// or: factory := func(i int) (*poly.VolumetricNetwork, error) { ... any arch ... }

master, err := poly.NeuralFountain(factory, batches, poly.DefaultNeuralFountainConfig())
out, err := master.Forward(inputTensor)
```

See `poly/neural_fountain.go`, `poly/weight_pack.go`, `poly/fountain_lt.go`.
