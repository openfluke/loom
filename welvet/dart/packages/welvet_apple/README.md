# welvet_apple

iOS and macOS natives for [`welvet`](https://pub.dev/packages/welvet). This tarball contains **only**:

```
ios/Welvet.xcframework/
macos/Frameworks/libwelvet.dylib
native/macos_universal/libwelvet.dylib   # VM / flutter test fallback
```

Do not add this package directly unless you need Apple-only artifacts. Flutter apps should depend on `welvet` — it pulls `welvet_apple` automatically on iOS and macOS.
