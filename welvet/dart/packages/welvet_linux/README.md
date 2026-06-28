# welvet_linux

Linux natives for [`welvet`](https://pub.dev/packages/welvet). This tarball contains **only**:

```
native/linux_amd64/libwelvet.so
native/linux_arm64/libwelvet.so
```

Do not add this package directly unless you need Linux-only artifacts. Flutter apps should depend on `welvet` — it pulls `welvet_linux` automatically on Linux.
