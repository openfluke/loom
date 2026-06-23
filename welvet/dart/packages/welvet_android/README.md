# welvet_android

Android natives for [`welvet`](https://pub.dev/packages/welvet). This tarball contains **only**:

```
native/android/arm64-v8a/libwelvet.so
native/android/x86_64/libwelvet.so
```

Do not add this package directly unless you need Android-only artifacts. Flutter apps should depend on `welvet` — it pulls `welvet_android` automatically on Android.
