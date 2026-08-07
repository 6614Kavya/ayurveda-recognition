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

    // ---------------------------------------------------------------
    // Module 2 (single leaf) result shape. Untouched.
    // ---------------------------------------------------------------
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

    // ---------------------------------------------------------------
    // Module 3 (compound leaf) result shape.
    // Backend now returns `symptoms` (list of {name, description,
    // group, percentage}) instead of `breakdown` (map of raw
    // `worst_*` column -> %). `breakdown` kept as a fallback only so
    // this doesn't break mid-rollout against an older backend build;
    // safe to delete the fallback branch once the new backend is the
    // only one in use.
    // ---------------------------------------------------------------
    final species = health['species']?.toString();
    final decision = health['decision']?.toString() ?? 'Unknown';
    final decisionConfidence = asFraction(health['decision_confidence']);
    final healthValue = health['health_value'];
    final severityRaw = health['severity_score_raw'];

    final symptomsRaw = health['symptoms'] as List?;
    final symptoms = symptomsRaw?.map((e) => (e as Map).cast<String, dynamic>()).toList();

    final legacyBreakdown = (health['breakdown'] as Map?)?.cast<String, dynamic>() ?? {};

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
        if (symptoms != null && symptoms.isNotEmpty) ...[
          const SizedBox(height: 16),
          const Text('What we found',
              style: TextStyle(fontWeight: FontWeight.w600, color: AppColors.sectionTitle, fontSize: 13)),
          const SizedBox(height: 6),
          _SymptomsList(symptoms: symptoms),
        ] else if (symptoms == null && legacyBreakdown.isNotEmpty) ...[
          // Fallback for an older backend still returning `breakdown`.
          const SizedBox(height: 16),
          const Text('Breakdown',
              style: TextStyle(fontWeight: FontWeight.w600, color: AppColors.sectionTitle, fontSize: 13)),
          const SizedBox(height: 6),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            decoration: BoxDecoration(color: AppColors.outputBg, borderRadius: BorderRadius.circular(8)),
            child: Column(
              children: legacyBreakdown.entries
                  .map((e) => KeyValueRow(label: titleCase(e.key), value: formatValue(e.value)))
                  .toList(),
            ),
          ),
        ],
      ],
    );
  }
}

/// Renders Module 3's `symptoms` list grouped by `group`, each item
/// showing its human name, one-line description, and raw percentage
/// (server-computed; no client-side relabeling or thresholds needed).
class _SymptomsList extends StatelessWidget {
  final List<Map<String, dynamic>> symptoms;
  const _SymptomsList({required this.symptoms});

  @override
  Widget build(BuildContext context) {
    final grouped = <String, List<Map<String, dynamic>>>{};
    for (final s in symptoms) {
      final group = s['group']?.toString() ?? 'Other';
      grouped.putIfAbsent(group, () => []).add(s);
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: grouped.entries.map((entry) {
        return Padding(
          padding: const EdgeInsets.only(bottom: 10),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Padding(
                padding: const EdgeInsets.only(bottom: 4),
                child: Text(
                  entry.key,
                  style: const TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w600,
                    color: AppColors.sectionTitle,
                  ),
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                decoration: BoxDecoration(
                  color: AppColors.outputBg,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Column(
                  children: entry.value.map((s) => _SymptomRow(symptom: s)).toList(),
                ),
              ),
            ],
          ),
        );
      }).toList(),
    );
  }
}

class _SymptomRow extends StatelessWidget {
  final Map<String, dynamic> symptom;
  const _SymptomRow({required this.symptom});

  @override
  Widget build(BuildContext context) {
    final name = symptom['name']?.toString() ?? '';
    final description = symptom['description']?.toString() ?? '';
    final pct = symptom['percentage'];
    final pctText = pct is num ? '${pct.toStringAsFixed(1)}%' : '—';

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(name, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w500)),
                if (description.isNotEmpty)
                  Padding(
                    padding: const EdgeInsets.only(top: 2),
                    child: Text(
                      description,
                      style: TextStyle(fontSize: 11.5, color: AppColors.sectionTitle.withOpacity(0.75)),
                    ),
                  ),
              ],
            ),
          ),
          const SizedBox(width: 10),
          Text(
            pctText,
            style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600),
          ),
        ],
      ),
    );
  }
}