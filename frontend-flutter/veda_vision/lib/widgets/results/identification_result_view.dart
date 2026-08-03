import 'package:flutter/material.dart';
import '../../theme/app_colors.dart';
import '../../utils/format_utils.dart';
import '../badge.dart';
import '../confidence_meter.dart';
import '../bullet_line.dart';

class IdentificationResultView extends StatelessWidget {
  final Map<String, dynamic> data;
  const IdentificationResultView({super.key, required this.data});

  List<String> _asStringList(dynamic v) {
    if (v == null) return [];
    if (v is List) return v.map((e) => e.toString()).toList();
    if (v is String && v.trim().isNotEmpty) return [v];
    return [];
  }

  String _friendlyModule(dynamic m) {
    switch (m) {
      case 'module2_single_leaves':
        return 'Simple Leaf';
      case 'module3_compound_leaves':
        return 'Compound Leaf';
      default:
        return m?.toString() ?? '';
    }
  }

  @override
  Widget build(BuildContext context) {
    final plantName = data['plant_name']?.toString() ?? 'Unknown';
    final sinhalaName = data['sinhala_name']?.toString();
    final module = data['module'];
    final confidence = asFraction(data['confidence']);
    final uses = _asStringList(data['uses']);
    final diseases = _asStringList(data['diseases_treated']);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('🌿 ', style: TextStyle(fontSize: 20)),
            Expanded(
              child: Text(plantName,
                  style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: AppColors.titleColor)),
            ),
          ],
        ),
        if (sinhalaName != null && sinhalaName.isNotEmpty)
          Padding(
            padding: const EdgeInsets.only(top: 2, left: 26),
            child: Text(sinhalaName,
                style: const TextStyle(fontStyle: FontStyle.italic, color: Colors.black54, fontSize: 13)),
          ),
        const SizedBox(height: 12),
        if (module != null) AppBadge(text: _friendlyModule(module), color: AppColors.sectionTitle),
        const SizedBox(height: 12),
        ConfidenceMeter(label: 'Prediction confidence', fraction: confidence),
        if (uses.isNotEmpty) ...[
          const SizedBox(height: 16),
          const Text('Traditional Uses',
              style: TextStyle(fontWeight: FontWeight.w600, color: AppColors.sectionTitle, fontSize: 13)),
          const SizedBox(height: 6),
          ...uses.map((u) => BulletLine(text: u)),
        ],
        if (diseases.isNotEmpty) ...[
          const SizedBox(height: 16),
          const Text('Diseases Treated',
              style: TextStyle(fontWeight: FontWeight.w600, color: AppColors.sectionTitle, fontSize: 13)),
          const SizedBox(height: 6),
          Wrap(
            spacing: 6,
            runSpacing: 6,
            children: diseases.map((d) => AppBadge(text: d, color: AppColors.buttonBg)).toList(),
          ),
        ],
      ],
    );
  }
}