import 'dart:convert';
import 'package:flutter/material.dart';
import '../output_panel.dart';
import 'identification_result_view.dart';
import 'leaf_health_result_view.dart';

/// Picks the right formatted view based on the shape of the response JSON.
/// Falls back to raw JSON so nothing is ever silently hidden.
class PredictionResultView extends StatelessWidget {
  final Map<String, dynamic> data;
  const PredictionResultView({super.key, required this.data});

  @override
  Widget build(BuildContext context) {
    if (data.containsKey('health_result')) {
      return LeafHealthResultView(data: data);
    }
    if (data.containsKey('plant_name')) {
      return IdentificationResultView(data: data);
    }
    return OutputPanel(text: const JsonEncoder.withIndent('  ').convert(data));
  }
}