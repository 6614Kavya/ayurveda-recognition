import 'package:flutter/material.dart';
import '../../theme/app_colors.dart';
import '../../utils/format_utils.dart';
import '../badge.dart';
import '../confidence_meter.dart';
import '../stat_tile.dart';
import '../key_value_row.dart';

class LeafHealthResultView extends StatelessWidget {
  final Map<String, dynamic> data;
  const LeafHealthResultView({super.key, required this.data});

  Color _decisionColor(String decision) {
    final d = decision.toLowerCase();
    if (d.contains('healthy') || d.contains('good')) return AppColors.buttonBg;
    if (d.contains('unhealthy') || d.contains('disease') || d.contains('infect') || d.contains('poor')) {
      return AppColors.errorColor;
    }
    return AppColors.amber;
  }

  @override
  Widget build(BuildContext context) {
    final leafType = data['leaf_type']?.toString() ?? '';
    final routeConfidence = asFraction(data['confidence']);
    final health = (data['health_result'] as Map?)?.cast<String, dynamic>() ?? {};
    final isSimpleStage = health.containsKey('stage1_status');

    final leafBadgeText = leafType.isEmpty ? 'Leaf' : '${leafType[0].toUpperCase()}${leafType.substring(1)} Leaf';

    if (isSimpleStage) {
      final decision = health['stage1_status']?.toString() ?? 'Unknown';
      final decisionConfidence = parsePercent(health['stage1_confidence']);
      final healthPercentage = health['health_percentage']?.toString() ?? '—';

      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              AppBadge(text: leafBadgeText, color: AppColors.sectionTitle),
              AppBadge(text: decision, color: _decisionColor(decision)),
            ],
          ),
          const SizedBox(height: 14),
          ConfidenceMeter(label: 'Leaf-type routing confidence', fraction: routeConfidence),
          const SizedBox(height: 10),
          ConfidenceMeter(label: 'Stage 1 confidence', fraction: decisionConfidence),
          const SizedBox(height: 16),
          StatTile(label: 'Health Percentage', value: healthPercentage),
        ],
      );
    }

    final species = health['species']?.toString();
    final decision = health['decision']?.toString() ?? 'Unknown';
    final decisionConfidence = asFraction(health['decision_confidence']);
    final healthValue = health['health_value'];
    final severityRaw = health['severity_score_raw'];
    final breakdown = (health['breakdown'] as Map?)?.cast<String, dynamic>() ?? {};

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            AppBadge(text: leafBadgeText, color: AppColors.sectionTitle),
            if (species != null) AppBadge(text: species, color: AppColors.titleColor),
            AppBadge(text: decision, color: _decisionColor(decision)),
          ],
        ),
        const SizedBox(height: 14),
        ConfidenceMeter(label: 'Leaf-type routing confidence', fraction: routeConfidence),
        const SizedBox(height: 10),
        ConfidenceMeter(label: 'Health decision confidence', fraction: decisionConfidence),
        const SizedBox(height: 16),
        Row(
          children: [
            Expanded(child: StatTile(label: 'Health Value', value: formatValue(healthValue))),
            const SizedBox(width: 12),
            if (severityRaw != null) ...[
            Expanded(child: StatTile(label: 'Severity Score', value: formatValue(severityRaw))),
            ],
          ],
        ),
        if (breakdown.isNotEmpty) ...[
          const SizedBox(height: 16),
          const Text('Breakdown',
              style: TextStyle(fontWeight: FontWeight.w600, color: AppColors.sectionTitle, fontSize: 13)),
          const SizedBox(height: 6),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            decoration: BoxDecoration(color: AppColors.outputBg, borderRadius: BorderRadius.circular(8)),
            child: Column(
              children: breakdown.entries
                  .map((e) => KeyValueRow(label: titleCase(e.key), value: formatValue(e.value)))
                  .toList(),
            ),
          ),
        ],
      ],
    );
  }
}