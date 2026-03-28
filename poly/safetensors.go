package poly

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"unsafe"
)

// SafetensorsHeader contains metadata about tensors in the file
type SafetensorsHeader struct {
	Tensors map[string]TensorInfo `json:"-"`
}

// TensorInfo describes a tensor's properties
type TensorInfo struct {
	DType  string `json:"dtype"`
	Shape  []int  `json:"shape"`
	Offset []int  `json:"data_offsets"`
}

// TensorWithShape holds tensor data along with its shape
type TensorWithShape struct {
	Values []float32
	Shape  []int
	DType  string
}

// LoadSafetensors reads a safetensors file and returns tensors by name
func LoadSafetensors(filepath string) (map[string][]float32, error) {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}
	return LoadSafetensorsFromBytes(data)
}

// LoadSafetensorsFromBytes reads safetensors data from a byte slice and returns tensors by name
func LoadSafetensorsFromBytes(data []byte) (map[string][]float32, error) {
	if len(data) < 8 {
		return nil, fmt.Errorf("data too short: need at least 8 bytes for header size")
	}

	headerSize := binary.LittleEndian.Uint64(data[0:8])
	if len(data) < int(8+headerSize) {
		return nil, fmt.Errorf("data too short: header size %d but only %d bytes available", headerSize, len(data)-8)
	}

	headerBytes := data[8 : 8+headerSize]
	var rawHeader map[string]interface{}
	if err := json.Unmarshal(headerBytes, &rawHeader); err != nil {
		return nil, fmt.Errorf("failed to parse header: %w", err)
	}

	allData := data[8+headerSize:]
	tensors := make(map[string][]float32)

	for name, value := range rawHeader {
		if name == "__metadata__" {
			continue
		}

		infoMap, ok := value.(map[string]interface{})
		if !ok {
			continue
		}

		encodedDtype, _ := infoMap["dtype"].(string)
		shapeList, _ := infoMap["shape"].([]interface{})
		offsetList, _ := infoMap["data_offsets"].([]interface{})

		shape := make([]int, len(shapeList))
		for i, v := range shapeList {
			shape[i] = int(v.(float64))
		}

		startOffset := int(offsetList[0].(float64))
		numElements := 1
		for _, dim := range shape {
			numElements *= dim
		}

		tensorData := make([]float32, numElements)

		if err := decodeTensorData(allData, startOffset, encodedDtype, numElements, tensorData); err != nil {
			return nil, fmt.Errorf("tensor %s: %w", name, err)
		}

		tensors[name] = tensorData
	}

	return tensors, nil
}

// TensorMetaHeader is used for JSON serialization of tensor metadata in safetensors files
type TensorMetaHeader struct {
	DType       string `json:"dtype"`
	Shape       []int  `json:"shape"`
	DataOffsets []int  `json:"data_offsets"`
}

// buildSafetensorsHeader is a shared helper to construct the header and calculate offsets.

// SaveSafetensorsToBytes returns the safetensors file as a byte slice.
func SaveSafetensorsToBytes(n *VolumetricNetwork) ([]byte, error) {
	tensors := make(map[string]TensorWithShape)
	for i, l := range n.Layers {
		if l.WeightStore != nil {
			name := fmt.Sprintf("layer_%d.weights", i)
			tensors[name] = TensorWithShape{
				Values: l.WeightStore.Master,
				Shape:  []int{len(l.WeightStore.Master)},
				DType:  "F32",
			}
		}
	}

	headerJSON, totalBytes, header, err := buildSafetensorsHeader(tensors)
	if err != nil {
		return nil, err
	}

	headerLen := uint64(len(headerJSON))
	out := make([]byte, 8+headerLen+totalBytes)

	binary.LittleEndian.PutUint64(out[0:8], headerLen)
	copy(out[8:8+headerLen], headerJSON)

	dataStart := 8 + headerLen
	for name, info := range header {
		t := tensors[name]
		offset := dataStart + uint64(info.DataOffsets[0])
		
		// Only supporting F32 for now in this export
		for j, v := range t.Values {
			binary.LittleEndian.PutUint32(out[offset+uint64(j*4):offset+uint64(j*4+4)], math.Float32bits(v))
		}
	}

	return out, nil
}

func buildSafetensorsHeader(tensors map[string]TensorWithShape) ([]byte, uint64, map[string]TensorMetaHeader, error) {
	header := make(map[string]TensorMetaHeader)
	var totalBytes uint64 = 0

	var names []string
	for k := range tensors {
		names = append(names, k)
	}

	for _, name := range names {
		t := tensors[name]
		numElements := 1
		for _, dim := range t.Shape {
			numElements *= dim
		}

		var bytesPerElement int
		switch t.DType {
		case "F32", "I32", "U32": bytesPerElement = 4
		case "F64", "I64", "U64": bytesPerElement = 8
		default: bytesPerElement = 4
		}

		size := uint64(numElements * bytesPerElement)
		header[name] = TensorMetaHeader{
			DType:       t.DType,
			Shape:       t.Shape,
			DataOffsets: []int{int(totalBytes), int(totalBytes + size)},
		}
		totalBytes += size
	}

	headerJSON, err := json.Marshal(header)
	if err != nil {
		return nil, 0, nil, fmt.Errorf("failed to marshal header: %w", err)
	}

	return headerJSON, totalBytes, header, nil
}

// SaveSafetensors saves a map of tensors to a safetensors file
func SaveSafetensors(filepath string, tensors map[string]TensorWithShape) error {
	headerJSON, totalBytes, header, err := buildSafetensorsHeader(tensors)
	if err != nil {
		return err
	}

	headerLen := uint64(len(headerJSON))

	f, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer f.Close()

	lenBuf := make([]byte, 8)
	binary.LittleEndian.PutUint64(lenBuf, headerLen)
	if _, err := f.Write(lenBuf); err != nil {
		return err
	}

	if _, err := f.Write(headerJSON); err != nil {
		return err
	}

	// We need deterministic order for the data section too
	var names []string
	for name := range header {
		names = append(names, name)
	}

	for _, name := range names {
		t := tensors[name]
		if t.DType == "F32" {
			dataBuf := make([]byte, len(t.Values)*4)
			for i, v := range t.Values {
				bits := math.Float32bits(v)
				binary.LittleEndian.PutUint32(dataBuf[i*4:], bits)
			}
			if _, err := f.Write(dataBuf); err != nil {
				return err
			}
		} else {
			return fmt.Errorf("only F32 saving supported currently")
		}
	}
	_ = totalBytes // keep for compatibility if needed elsewhere
	return nil
}

// LoadSafetensorsWithShapes loads safetensors and returns both values and shapes
func LoadSafetensorsWithShapes(data []byte) (map[string]TensorWithShape, error) {
	if len(data) < 8 {
		return nil, fmt.Errorf("data too short: need at least 8 bytes for header size")
	}

	headerSize := binary.LittleEndian.Uint64(data[0:8])
	if len(data) < int(8+headerSize) {
		return nil, fmt.Errorf("data too short: header size %d but only %d bytes available", headerSize, len(data)-8)
	}

	headerBytes := data[8 : 8+headerSize]
	var rawHeader map[string]interface{}
	if err := json.Unmarshal(headerBytes, &rawHeader); err != nil {
		return nil, fmt.Errorf("failed to parse header: %w", err)
	}

	allData := data[8+headerSize:]
	tensors := make(map[string]TensorWithShape)

	for name, value := range rawHeader {
		if name == "__metadata__" {
			continue
		}

		infoMap, ok := value.(map[string]interface{})
		if !ok {
			continue
		}

		encodedDtype, _ := infoMap["dtype"].(string)
		shapeList, _ := infoMap["shape"].([]interface{})
		offsetList, _ := infoMap["data_offsets"].([]interface{})

		shape := make([]int, len(shapeList))
		for i, v := range shapeList {
			shape[i] = int(v.(float64))
		}

		startOffset := int(offsetList[0].(float64))
		numElements := 1
		for _, dim := range shape {
			numElements *= dim
		}

		tensorData := make([]float32, numElements)
		if err := decodeTensorData(allData, startOffset, encodedDtype, numElements, tensorData); err != nil {
			fmt.Printf("Warning: skipping tensor %s: %v\n", name, err)
			continue
		}

		tensors[name] = TensorWithShape{
			Values: tensorData,
			Shape:  shape,
			DType:  encodedDtype,
		}
	}

	return tensors, nil
}

func decodeTensorData(allData []byte, startOffset int, dtype string, numElements int, out []float32) error {
	switch dtype {
	case "F32":
		for i := 0; i < numElements; i++ {
			offset := startOffset + i*4
			if offset+4 > len(allData) {
				return fmt.Errorf("data out of bounds")
			}
			bits := binary.LittleEndian.Uint32(allData[offset : offset+4])
			out[i] = *(*float32)(unsafe.Pointer(&bits))
		}
	case "F64":
		for i := 0; i < numElements; i++ {
			offset := startOffset + i*8
			if offset+8 > len(allData) {
				return fmt.Errorf("data out of bounds")
			}
			bits := binary.LittleEndian.Uint64(allData[offset : offset+8])
			out[i] = float32(math.Float64frombits(bits))
		}
	case "F16":
		for i := 0; i < numElements; i++ {
			offset := startOffset + i*2
			if offset+2 > len(allData) {
				return fmt.Errorf("data out of bounds")
			}
			f16bits := binary.LittleEndian.Uint16(allData[offset : offset+2])
			out[i] = float16ToFloat32(f16bits)
		}
	case "BF16":
		for i := 0; i < numElements; i++ {
			offset := startOffset + i*2
			if offset+2 > len(allData) {
				return fmt.Errorf("data out of bounds")
			}
			bf16bits := binary.LittleEndian.Uint16(allData[offset : offset+2])
			out[i] = bfloat16ToFloat32(bf16bits)
		}
	case "F4":
		for i := 0; i < numElements; i++ {
			byteOffset := startOffset + i/2
			if byteOffset >= len(allData) {
				return fmt.Errorf("data out of bounds")
			}
			b := allData[byteOffset]
			var fp4bits uint8
			if i%2 == 0 {
				fp4bits = b & 0x0F
			} else {
				fp4bits = (b >> 4) & 0x0F
			}
			out[i] = fp4ToFloat32(fp4bits)
		}
	case "I8":
		for i := 0; i < numElements; i++ {
			offset := startOffset + i
			if offset >= len(allData) {
				return fmt.Errorf("data out of bounds")
			}
			out[i] = float32(int8(allData[offset]))
		}
	case "I16":
		for i := 0; i < numElements; i++ {
			offset := startOffset + i*2
			if offset+2 > len(allData) {
				return fmt.Errorf("data out of bounds")
			}
			val := int16(binary.LittleEndian.Uint16(allData[offset : offset+2]))
			out[i] = float32(val)
		}
	case "I32":
		for i := 0; i < numElements; i++ {
			offset := startOffset + i*4
			if offset+4 > len(allData) {
				return fmt.Errorf("data out of bounds")
			}
			val := int32(binary.LittleEndian.Uint32(allData[offset : offset+4]))
			out[i] = float32(val)
		}
	case "I64":
		for i := 0; i < numElements; i++ {
			offset := startOffset + i*8
			if offset+8 > len(allData) {
				return fmt.Errorf("data out of bounds")
			}
			val := int64(binary.LittleEndian.Uint64(allData[offset : offset+8]))
			out[i] = float32(val)
		}
	case "U8":
		for i := 0; i < numElements; i++ {
			offset := startOffset + i
			if offset >= len(allData) {
				return fmt.Errorf("data out of bounds")
			}
			out[i] = float32(allData[offset])
		}
	case "U16":
		for i := 0; i < numElements; i++ {
			offset := startOffset + i*2
			if offset+2 > len(allData) {
				return fmt.Errorf("data out of bounds")
			}
			val := binary.LittleEndian.Uint16(allData[offset : offset+2])
			out[i] = float32(val)
		}
	case "U32":
		for i := 0; i < numElements; i++ {
			offset := startOffset + i*4
			if offset+4 > len(allData) {
				return fmt.Errorf("data out of bounds")
			}
			val := binary.LittleEndian.Uint32(allData[offset : offset+4])
			out[i] = float32(val)
		}
	case "U64":
		for i := 0; i < numElements; i++ {
			offset := startOffset + i*8
			if offset+8 > len(allData) {
				return fmt.Errorf("data out of bounds")
			}
			val := binary.LittleEndian.Uint64(allData[offset : offset+8])
			out[i] = float32(val)
		}
	default:
		return fmt.Errorf("unsupported dtype: %s", dtype)
	}
	return nil
}

func float16ToFloat32(f16 uint16) float32 {
	sign := uint32((f16 >> 15) & 0x1)
	exponent := uint32((f16 >> 10) & 0x1F)
	mantissa := uint32(f16 & 0x3FF)
	var f32bits uint32
	if exponent == 0 {
		if mantissa == 0 {
			f32bits = sign << 31
		} else {
			exponent = 1
			for (mantissa & 0x400) == 0 {
				mantissa <<= 1
				exponent--
			}
			mantissa &= 0x3FF
			f32bits = (sign << 31) | ((exponent + (127 - 15)) << 23) | (mantissa << 13)
		}
	} else if exponent == 0x1F {
		f32bits = (sign << 31) | (0xFF << 23) | (mantissa << 13)
	} else {
		f32bits = (sign << 31) | ((exponent + (127 - 15)) << 23) | (mantissa << 13)
	}
	return *(*float32)(unsafe.Pointer(&f32bits))
}

func bfloat16ToFloat32(bf16 uint16) float32 {
	f32bits := uint32(bf16) << 16
	return *(*float32)(unsafe.Pointer(&f32bits))
}

func fp4ToFloat32(fp4 uint8) float32 {
	sign := uint32((fp4 >> 3) & 0x1)
	exponent := uint32((fp4 >> 1) & 0x3)
	mantissa := uint32(fp4 & 0x1)
	if exponent == 0 {
		if mantissa == 0 {
			bits := sign << 31
			return *(*float32)(unsafe.Pointer(&bits))
		}
		f32bits := (sign << 31) | (126 << 23) | (mantissa << 22)
		return *(*float32)(unsafe.Pointer(&f32bits))
	} else if exponent == 3 {
		f32bits := (sign << 31) | (0xFF << 23) | (mantissa << 22)
		return *(*float32)(unsafe.Pointer(&f32bits))
	} else {
		exp_f32 := (exponent - 1) + 127
		f32bits := (sign << 31) | (exp_f32 << 23) | (mantissa << 22)
		return *(*float32)(unsafe.Pointer(&f32bits))
	}
}
