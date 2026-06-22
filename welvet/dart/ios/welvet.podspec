#
# Welvet FFI plugin — iOS (static Welvet.xcframework + force link for dlsym).
#
Pod::Spec.new do |s|
  s.name             = 'welvet'
  s.version          = '0.80.4'
  s.summary          = 'Loom Welvet C-ABI bindings for Flutter/Dart'
  s.description      = <<-DESC
Flutter FFI plugin bundling prebuilt Welvet (Loom M-POLY-VTD) natives.
                       DESC
  s.homepage         = 'https://github.com/openfluke/loom'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'OpenFluke' => 'https://github.com/openfluke' }
  s.source           = { :path => '.' }
  s.source_files     = 'Classes/welvet_force_link.c'
  s.vendored_frameworks = 'Welvet.xcframework'
  s.dependency 'Flutter'
  s.platform = :ios, '13.0'
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386',
    'OTHER_LDFLAGS' => '-force_load $(PODS_TARGET_SRCROOT)/Welvet.xcframework/ios-arm64/libwelvet.a',
  }
  s.swift_version = '5.0'
end
