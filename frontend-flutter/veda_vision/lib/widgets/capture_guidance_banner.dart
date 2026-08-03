import 'package:flutter/material.dart';
import '../theme/app_colors.dart';

enum CaptureSubject { flower, leaf }

/// A small instructional strip shown above each upload control, telling the
/// user what background to shoot against. This directly reflects your
/// preprocessing pipeline's assumptions (plain, high-contrast background)
/// so uploaded photos actually segment cleanly on the backend.
class CaptureGuidanceBanner extends StatelessWidget {
  final CaptureSubject subject;
  const CaptureGuidanceBanner({super.key, required this.subject});

  String get _text {
    switch (subject) {
      case CaptureSubject.flower:
        return 'Place the flower against a plain black or white background, '
            'in good even light, for the most accurate result.';
      case CaptureSubject.leaf:
        return 'Lay the leaf flat on a plain white background, '
            'with the whole leaf visible in the frame.';
    }
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      decoration: BoxDecoration(
        color: AppColors.infoBg,
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: AppColors.infoBorder),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.info_outline, size: 15, color: AppColors.infoText),
          const SizedBox(width: 6),
          Expanded(
            child: Text(
              _text,
              style: const TextStyle(
                fontSize: 11.5,
                color: AppColors.infoText,
                height: 1.35,
              ),
            ),
          ),
        ],
      ),
    );
  }
}