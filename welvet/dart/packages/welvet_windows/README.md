# welvet_windows

Windows natives for [`welvet`](https://pub.dev/packages/welvet). This tarball contains **only**:

```
native/windows_amd64/welvet.dll
native/windows_arm64/welvet.dll
```

Do not add this package directly unless you need Windows-only artifacts. Flutter apps should depend on `welvet` — it pulls `welvet_windows` automatically on Windows.
