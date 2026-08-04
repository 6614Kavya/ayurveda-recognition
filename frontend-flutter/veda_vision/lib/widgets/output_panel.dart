import 'package:flutter/material.dart';
import '../theme/app_colors.dart';

class OutputPanel extends StatelessWidget {
  final String text;
  const OutputPanel({super.key, required this.text});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(color: AppColors.outputBg, borderRadius: BorderRadius.circular(10)),
      child: Text(
        text,
        style: const TextStyle(
          fontFamily: 'monospace',
          fontSize: 13,
          color: AppColors.textPrimary,
          height: 1.5,
        ),
      ),
    );
  }
}