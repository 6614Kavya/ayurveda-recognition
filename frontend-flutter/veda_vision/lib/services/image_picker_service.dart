import 'dart:io' show Platform;
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:image_picker/image_picker.dart';
import 'package:file_selector/file_selector.dart' as fs;
import '../models/picked_image.dart';

class ImagePickerService {
  static bool get isDesktop =>
      !kIsWeb && (Platform.isWindows || Platform.isLinux || Platform.isMacOS);

  /// Picks an image from camera or gallery. On desktop, gallery picks route
  /// through file_selector instead of image_picker, since image_picker's
  /// desktop gallery filter hardcodes extensions and excludes HEIC/HEIF.
  static Future<PickedImage?> pick(ImageSource source) async {
    if (isDesktop && source == ImageSource.gallery) {
      final typeGroup = fs.XTypeGroup(
        label: 'images',
        extensions: ['jpg', 'jpeg', 'png', 'bmp', 'webp', 'heic', 'heif'],
      );
      final file = await fs.openFile(acceptedTypeGroups: [typeGroup]);
      if (file == null) return null;
      final bytes = await file.readAsBytes();
      return PickedImage(bytes: bytes, filename: file.name);
    }

    final picker = ImagePicker();
    final picked = await picker.pickImage(source: source);
    if (picked == null) return null;
    final bytes = await picked.readAsBytes();
    return PickedImage(bytes: bytes, filename: picked.name);
  }
}