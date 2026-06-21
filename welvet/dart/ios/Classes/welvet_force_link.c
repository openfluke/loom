/* Ensures libwelvet.a (Welvet.xcframework) stays linked for Dart FFI dlsym. */
extern void FreeLoomString(char *ptr);

__attribute__((used)) static void (*const _welvet_force_link)(char *) = FreeLoomString;
