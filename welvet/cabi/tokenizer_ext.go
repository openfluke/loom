package main

/*
#include <stdlib.h>
#include <stdint.h>
*/
import "C"

import (
	"encoding/json"

	"github.com/openfluke/loom/poly"
)

//export LoomLoadTokenizer
func LoomLoadTokenizer(path *C.char) C.longlong {
	p := C.GoString(path)
	t, err := poly.LoadTokenizer(p)
	if err != nil {
		return -1
	}

	networkMu.Lock()
	id := tokenizerNextID
	tokenizerNextID++
	tokenizers[id] = t
	networkMu.Unlock()

	return C.longlong(id)
}

//export LoomTokenize
func LoomTokenize(handle C.longlong, text *C.char) *C.char {
	networkMu.RLock()
	t, ok := tokenizers[int64(handle)]
	networkMu.RUnlock()
	if !ok {
		return errJSON("invalid tokenizer handle")
	}

	str := C.GoString(text)
	tokens := t.Encode(str, true)

	data, _ := json.Marshal(tokens)
	return C.CString(string(data))
}

//export LoomDetokenize
func LoomDetokenize(handle C.longlong, tokensJSON *C.char) *C.char {
	networkMu.RLock()
	t, ok := tokenizers[int64(handle)]
	networkMu.RUnlock()
	if !ok {
		return errJSON("invalid tokenizer handle")
	}

	var ids []uint32
	if err := json.Unmarshal([]byte(C.GoString(tokensJSON)), &ids); err != nil {
		return errJSON("invalid tokens JSON")
	}

	text := t.Decode(ids, true)
	return C.CString(text)
}

//export LoomFreeTokenizer
func LoomFreeTokenizer(handle C.longlong) {
	networkMu.Lock()
	delete(tokenizers, int64(handle))
	networkMu.Unlock()
}
