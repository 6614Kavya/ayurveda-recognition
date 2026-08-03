import 'package:flutter/material.dart';
import '../theme/app_colors.dart';

class BulletLine extends StatelessWidget {
  final String text;
  const BulletLine({super.key, required this.text});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('•  ', style: TextStyle(color: AppColors.buttonBg, fontWeight: FontWeight.bold)),
          Expanded(child: Text(text, style: const TextStyle(fontSize: 13, color: AppColors.textPrimary, height: 1.4))),
        ],
      ),
    );
  }
}