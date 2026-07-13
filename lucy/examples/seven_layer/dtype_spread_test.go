package sevenlayer

import (
	"testing"
	"time"
)

func TestDtypeSpreadSlowestFastest(t *testing.T) {
	rows := []crossPathRow{
		{DType: "Float64", fwdSimd: 30 * time.Millisecond, natFwdSimd: 28 * time.Millisecond},
		{DType: "Int32", fwdSimd: 40 * time.Millisecond, natFwdSimd: 29 * time.Millisecond},
		{DType: "Uint32", fwdSimd: 37 * time.Millisecond, natFwdSimd: 27 * time.Millisecond},
	}
	slow, fast, ok := slowestFastestDtypes(rows, bestSimdDuelFwd)
	if !ok {
		t.Fatal("expected spread")
	}
	if slow.dtype != "Int32" || fast.dtype != "Uint32" {
		t.Fatalf("got slow=%s fast=%s, want Int32/Uint32", slow.dtype, fast.dtype)
	}
	if fast.d != 27*time.Millisecond {
		t.Fatalf("fastest fwd: got %v want 27ms", fast.d)
	}
	p := pairFromDtypeSpread(slow, fast)
	if p.ratio() == "—" {
		t.Fatal("expected ratio")
	}
}
