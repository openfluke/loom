#
# welvet_apple — macOS libwelvet.dylib (federated implementation of welvet).
#
Pod::Spec.new do |s|
  s.name             = 'welvet_apple'
  s.version          = '0.80.6'
  s.summary          = 'macOS natives for welvet (Loom Welvet C-ABI)'
  s.description      = <<-DESC
Federated Flutter FFI plugin — macOS libwelvet.dylib for package:welvet.
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
