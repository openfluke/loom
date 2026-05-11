//go:build android

package poly

/*
#include <android/log.h>
#include <stdlib.h>
void soulglitch_log(const char* msg) {
    __android_log_print(ANDROID_LOG_INFO, "SOULGLITCH", "%s", msg);
}
*/
import "C"
import (
	"unsafe"
)

func Alog(msg string) {
	cmsg := C.CString(msg)
	defer C.free(unsafe.Pointer(cmsg))
	C.soulglitch_log(cmsg)
}
