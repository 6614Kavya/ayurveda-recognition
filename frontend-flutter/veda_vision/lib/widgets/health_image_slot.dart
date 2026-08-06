import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../models/picked_image.dart';
import '../theme/app_colors.dart';
import 'app_button.dart';
<<<<<<< HEAD

/// Note: no guidance banner here anymore — it's shown once, above both
/// slots, by the parent section, so Top and Bottom always render at
/// identical height regardless of picked state.
=======
import 'capture_guidance_banner.dart';

>>>>>>> 747ae86bc245713dcdffcc52c4b501d1c45f63e4
class HealthImageSlot extends StatelessWidget {
  final String emoji;
  final String label;
  final PickedImage? picked;
  final void Function(ImageSource) onPick;
<<<<<<< HEAD
=======
  final CaptureSubject? guidance;
>>>>>>> 747ae86bc245713dcdffcc52c4b501d1c45f63e4

  const HealthImageSlot({
    super.key,
    required this.emoji,
    required this.label,
    required this.picked,
    required this.onPick,
<<<<<<< HEAD
=======
    this.guidance,
>>>>>>> 747ae86bc245713dcdffcc52c4b501d1c45f63e4
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
<<<<<<< HEAD
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 48,
            height: 48,
            child: picked != null
                ? ClipRRect(
                    borderRadius: BorderRadius.circular(6),
                    child: Image.memory(picked!.bytes, fit: BoxFit.cover),
                  )
                : Container(
                    decoration: BoxDecoration(
                      color: AppColors.outputBg,
                      borderRadius: BorderRadius.circular(6),
                    ),
                    alignment: Alignment.center,
                    child: Text(emoji, style: const TextStyle(fontSize: 20)),
                  ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Flexible(
                      child: Text(
                        '$emoji $label',
                        style: const TextStyle(
                          fontWeight: FontWeight.w500,
                          color: AppColors.textPrimary,
                          fontSize: 14,
                        ),
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
=======
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
>>>>>>> 747ae86bc245713dcdffcc52c4b501d1c45f63e4
          ),
        ],
      ),
    );
  }
}