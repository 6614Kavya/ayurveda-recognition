import 'package:flutter/foundation.dart' show Uint8List;

class PickedImage {
  final Uint8List bytes;
  final String filename;
  const PickedImage({required this.bytes, required this.filename});
}