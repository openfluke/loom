#
# Welvet FFI plugin — macOS (prebuilt libwelvet.dylib).
#
Pod::Spec.new do |s|
  s.name             = 'welvet'
  s.version          = '0.80.0'
  s.summary          = 'Loom Welvet C-ABI bindings for Flutter/Dart'
  s.description      = <<-DESC
Flutter FFI plugin bundling prebuilt Welvet (Loom M-POLY-VTD) natives.
                       DESC
  s.homepage         = 'https://github.com/openfluke/loom'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'OpenFluke' => 'https://github.com/openfluke' }
  s.source           = { :path => '.' }
  s.source_files     = 'Classes/**/*'
  s.vendored_libraries = 'Frameworks/libwelvet.dylib'
  s.dependency 'FlutterMacOS'
  s.platform = :osx, '10.14'
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES' }
  s.swift_version = '5.0'
end
