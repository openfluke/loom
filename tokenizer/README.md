# LOOM Tokenizer

Pure Go implementation of BPE (Byte Pair Encoding) tokenizer compatible with HuggingFace tokenizer.json format.

## Features

- ✅ Pure Go - no native dependencies
- ✅ Loads HuggingFace `tokenizer.json` format
- ✅ BPE encoding algorithm
- ✅ Special tokens support
- ✅ Byte fallback for unknown characters
- ✅ Compatible with Qwen, Llama, GPT-2, and other BPE-based models

## Usage

### Load from File

```go
import "github.com/openfluke/loom/tokenizer"

// Load tokenizer from file
tk, err := tokenizer.LoadFromFile("path/to/tokenizer.json")
if err != nil {
    log.Fatal(err)
}

// Encode text to token IDs
text := "Hello, world!"
tokens := tk.Encode(text, false)
fmt.Printf("Tokens: %v\n", tokens)

// Decode token IDs back to text
decoded := tk.Decode(tokens, false)
fmt.Printf("Decoded: %s\n", decoded)
```

### Load from Bytes (Recommended)

```go
import (
    "os"
    "github.com/openfluke/loom/tokenizer"
)

// Read tokenizer data into memory
data, err := os.ReadFile("path/to/tokenizer.json")
if err != nil {
    log.Fatal(err)
}

// Load tokenizer from bytes
tk, err := tokenizer.LoadFromBytes(data)
if err != nil {
    log.Fatal(err)
}

// Use tokenizer
tokens := tk.Encode("Hello, world!", false)
decoded := tk.Decode(tokens, false)

// Get vocabulary size
fmt.Printf("Vocab size: %d\n", tk.VocabSize())
```

**Why use LoadFromBytes?**

- Works with embedded data
- Load from network streams
- Custom storage backends (databases, cloud storage)
- Better for testing with mock data

## Supported Models

- Qwen / Qwen2.5 (BPE)
- Llama / Llama2 (BPE)
- GPT-2 / GPT-3 (BPE)
- Mistral (BPE)
- Most HuggingFace models using BPE tokenization

## Architecture

### BPE Algorithm

1. **Pre-tokenization**: Split text into words using regex patterns
2. **Character splitting**: Break words into individual characters
3. **Merge application**: Apply BPE merges in rank order
4. **Vocabulary lookup**: Convert final tokens to IDs
5. **Byte fallback**: Handle unknown tokens as raw bytes

### File Format

Compatible with HuggingFace's `tokenizer.json`:

```json
{
  "model": {
    "type": "BPE",
    "vocab": { "token": id, ... },
    "merges": ["first second", ...]
  },
  "added_tokens": [
    { "id": 151643, "content": "<|endoftext|>", "special": true }
  ]
}
```

## Extending

To add support for new tokenizer types:

1. Implement the tokenization algorithm in a new file (e.g., `wordpiece.go`, `unigram.go`)
2. Add a type field to detect the tokenizer type
3. Create a factory function in `bpe.go` to route to the correct implementation

Example:

```go
func LoadFromFile(path string) (*Tokenizer, error) {
    // Parse JSON
    var tokJSON TokenizerJSON
    // ...

    switch tokJSON.Model.Type {
    case "BPE":
        return loadBPE(tokJSON)
    case "WordPiece":
        return loadWordPiece(tokJSON)
    case "Unigram":
        return loadUnigram(tokJSON)
    default:
        return nil, fmt.Errorf("unsupported tokenizer type: %s", tokJSON.Model.Type)
    }
}
```

## Performance

- Encoding: ~1-2ms for 100 tokens
- Decoding: <1ms for 100 tokens
- Memory: Vocab size \* ~100 bytes

## Testing

```bash
cd tokenizer
go test -v
```

## Future Improvements

- [ ] WordPiece tokenizer (BERT)
- [ ] Unigram tokenizer (SentencePiece)
- [ ] Caching for frequently used merges
- [ ] Parallel encoding for long texts
- [ ] Character offset tracking
- [ ] Normalization (lowercase, unicode, etc.)
