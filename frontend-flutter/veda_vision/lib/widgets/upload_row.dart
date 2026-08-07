import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../theme/app_colors.dart';
import 'app_button.dart';
import 'capture_guidance_banner.dart';

class UploadRow extends StatelessWidget {
  final String emoji;
  final String label;
  final void Function(ImageSource) onPick;
  final CaptureSubject? guidance;

  const UploadRow({
    super.key,
    required this.emoji,
    required this.label,
    required this.onPick,
    this.guidance,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        border: Border.all(color: AppColors.inputBorder),
        borderRadius: BorderRadius.circular(6),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            '$emoji $label',
            style: const TextStyle(
              fontWeight: FontWeight.w500,
              color: AppColors.textPrimary,
              fontSize: 14,
            ),
          ),
          const SizedBox(height: 8),
          if (guidance != null) CaptureGuidanceBanner(subject: guidance!),
          Row(
            children: [
              AppButton(label: '📷 Camera', onPressed: () => onPick(ImageSource.camera)),
              const SizedBox(width: 8),
              AppButton(label: '🖼 Gallery', onPressed: () => onPick(ImageSource.gallery)),
            ],
          ),
        ],
      ),
    );
  }
}