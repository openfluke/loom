package poly

import "testing"

func TestMetaspaceTokenizerAddsBOSAndDecodes(t *testing.T) {
	data := []byte(`{
		"model": {
			"type": "BPE",
			"byte_fallback": true,
			"vocab": {
				"<unk>": 0,
				"<s>": 1,
				"▁": 2,
				"h": 3,
				"e": 4,
				"l": 5,
				"o": 6,
				"▁h": 7,
				"▁he": 8,
				"▁hel": 9,
				"▁hell": 10,
				"▁hello": 11
			},
			"merges": ["▁ h", "▁h e", "▁he l", "▁hel l", "▁hell o"]
		},
		"added_tokens": [
			{"id": 1, "content": "<s>", "special": true}
		],
		"normalizer": {
			"type": "Sequence",
			"normalizers": [
				{"type": "Prepend", "prepend": "▁"},
				{"type": "Replace", "pattern": {"String": " "}, "content": "▁"}
			]
		},
		"pre_tokenizer": null,
		"post_processor": {
			"type": "TemplateProcessing",
			"single": [{"SpecialToken": {"id": "<s>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
			"special_tokens": {"<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]}}
		}
	}`)

	tk, err := NewTokenizerFromJSON(data)
	if err != nil {
		t.Fatal(err)
	}
	got := tk.Encode("hello", true)
	want := []uint32{1, 11}
	if len(got) != len(want) {
		t.Fatalf("encoded len = %d, want %d (%v)", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("encoded[%d] = %d, want %d (all %v)", i, got[i], want[i], got)
		}
	}
	if decoded := tk.Decode(got, true); decoded != "hello" {
		t.Fatalf("decoded = %q, want hello", decoded)
	}
}
