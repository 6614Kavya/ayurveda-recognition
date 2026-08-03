import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../models/picked_image.dart';
import '../theme/app_colors.dart';
import 'app_button.dart';
import 'capture_guidance_banner.dart';

class HealthImageSlot extends StatelessWidget {
  final String emoji;
  final String label;
  final PickedImage? picked;
  final void Function(ImageSource) onPick;
  final CaptureSubject? guidance;

  const HealthImageSlot({
    super.key,
    required this.emoji,
    required this.label,
    required this.picked,
    required this.onPick,
    this.guidance,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        border: Border.all(
          color: picked != null ? AppColors.buttonBg : AppColors.inputBorder,
          width: picked != null ? 1.5 : 1,
        ),
        borderRadius: BorderRadius.circular(6),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (guidance != null) CaptureGuidanceBanner(subject: guidance!),
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              if (picked != null)
                ClipRRect(
                  borderRadius: BorderRadius.circular(6),
                  child: Image.memory(picked!.bytes, width: 48, height: 48, fit: BoxFit.cover),
                )
              else
                Container(
                  width: 48,
                  height: 48,
                  decoration: BoxDecoration(
                    color: AppColors.outputBg,
                    borderRadius: BorderRadius.circular(6),
                  ),
                  alignment: Alignment.center,
                  child: Text(emoji, style: const TextStyle(fontSize: 20)),
                ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Text(
                          '$emoji $label',
                          style: const TextStyle(
                            fontWeight: FontWeight.w500,
                            color: AppColors.textPrimary,
                            fontSize: 14,
                          ),
                        ),
                        if (picked != null) ...[
                          const SizedBox(width: 6),
                          const Icon(Icons.check_circle, color: AppColors.buttonBg, size: 16),
                        ],
                      ],
                    ),
                    if (picked != null)
                      Padding(
                        padding: const EdgeInsets.only(top: 2),
                        child: Text(
                          picked!.filename,
                          style: const TextStyle(fontSize: 11, color: Colors.black54),
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                    const SizedBox(height: 8),
                    Row(
                      children: [
                        AppButton(label: '📷 Camera', onPressed: () => onPick(ImageSource.camera)),
                        const SizedBox(width: 8),
                        AppButton(label: '🖼 Gallery', onPressed: () => onPick(ImageSource.gallery)),
                      ],
                    ),
                  ],
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}