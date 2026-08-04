import 'package:flutter/material.dart';
import '../theme/app_colors.dart';

class ConfidenceMeter extends StatelessWidget {
  final String label;
  final double fraction;
  const ConfidenceMeter({super.key, required this.label, required this.fraction});

  Color get _color {
    if (fraction >= 0.8) return AppColors.buttonBg;
    if (fraction >= 0.5) return AppColors.amber;
    return AppColors.errorColor;
  }

  @override
  Widget build(BuildContext context) {
    final pct = (fraction.clamp(0.0, 1.0) * 100).toStringAsFixed(1);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(label, style: const TextStyle(fontSize: 12, color: AppColors.textPrimary, fontWeight: FontWeight.w500)),
            Text('$pct%', style: TextStyle(fontSize: 12, color: _color, fontWeight: FontWeight.bold)),
          ],
        ),
        const SizedBox(height: 4),
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: LinearProgressIndicator(
            value: fraction.clamp(0.0, 1.0),
            minHeight: 8,
            backgroundColor: AppColors.inputBorder.withOpacity(0.4),
            valueColor: AlwaysStoppedAnimation<Color>(_color),
          ),
        ),
      ],
    );
  }
}