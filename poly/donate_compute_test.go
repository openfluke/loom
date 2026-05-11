package poly

import (
	"fmt"
	"testing"
	"time"
)

func TestDonateComputeModelPushRoundTrip(t *testing.T) {
	ln2, err := ServeDonateComputeTCP(DonateComputeServerOptions{Addr: "127.0.0.1:0", Mode: DonateServerModelPush, QueueCapacity: 8})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ln2.Close() })
	time.Sleep(20 * time.Millisecond)
	addr := ln2.Addr().String()

	cl, hi, err := DialDonateCompute(addr)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = cl.Close() })
	if hi == nil || hi.Type != DonateMsgHello {
		t.Fatalf("hello: %+v", hi)
	}
	if hi.Mode != string(DonateServerModelPush) {
		t.Fatalf("mode %q", hi.Mode)
	}

	if err := cl.PutModel(`{"ok":true}`, []byte{1, 2, 3, 4}); err != nil {
		t.Fatal(err)
	}
	res, err := cl.EnqueueInfer("j1", []int32{9, 9}, 3)
	if err != nil {
		t.Fatal(err)
	}
	if !res.OK || res.JobID != "j1" {
		t.Fatalf("infer: %+v", res)
	}
	if len(res.OutputIDs) < 2 {
		t.Fatalf("expected stub output: %+v", res.OutputIDs)
	}
}

func TestDonateComputeLocalLM(t *testing.T) {
	ln2, err := ServeDonateComputeTCP(DonateComputeServerOptions{
		Addr: "127.0.0.1:0", Mode: DonateServerLocalLM, LocalLmPath: "/tmp/model", QueueCapacity: 4,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ln2.Close() })
	time.Sleep(20 * time.Millisecond)
	addr := ln2.Addr().String()

	cl, _, err := DialDonateCompute(addr)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = cl.Close() })

	res, err := cl.EnqueuePrompt("p1", "hello world", 0)
	if err != nil {
		t.Fatal(err)
	}
	if !res.OK || res.Text == "" {
		t.Fatalf("prompt: %+v", res)
	}
}

func TestDonateComputeWrongMode(t *testing.T) {
	ln2, err := ServeDonateComputeTCP(DonateComputeServerOptions{Addr: "127.0.0.1:0", Mode: DonateServerLocalLM})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ln2.Close() })
	time.Sleep(20 * time.Millisecond)
	addr := ln2.Addr().String()

	cl, _, err := DialDonateCompute(addr)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = cl.Close() })

	_, err = cl.EnqueueInfer("x", []int32{1}, 1)
	if err == nil {
		t.Fatal("expected error for infer on local_lm")
	}
	if fmt.Sprint(err) == "" {
		t.Fatal("empty error")
	}
}
