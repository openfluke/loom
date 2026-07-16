package poly

import (
	"strings"
	"testing"
)

func TestReplyLooksDegenerate(t *testing.T) {
	ok := "I'm fine. Just trying to help out! How can I assist you today?"
	if ReplyLooksDegenerate(ok) {
		t.Fatalf("clean reply flagged: %q", ok)
	}
	junk := "everwastewaygroundwill happen thisï¼againstyardWISEBAOULDWAULEAVEGBATHugeryawaybackwardArgumentable"
	if !ReplyLooksDegenerate(junk) {
		t.Fatalf("junk reply not flagged: %q", junk)
	}
	spacedJunk := "I happenâÂ ðUS.--Union Turpunda -Trustroom Bux Buggenes Ves Kapuda Puckuts Punctures"
	if !ReplyLooksDegenerate(spacedJunk) {
		t.Fatalf("spaced mojibake junk not flagged: %q", spacedJunk)
	}
	san := SanitizeChatReply("Hello there. This is fine. " + junk)
	if san == "" || ReplyLooksDegenerate(san) {
		t.Fatalf("sanitize failed: %q", san)
	}
	if !strings.Contains(san, "This is fine.") {
		t.Fatalf("sanitize dropped clean sentence: %q", san)
	}
}
