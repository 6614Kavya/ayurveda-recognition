import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../models/picked_image.dart';
import '../services/api_service.dart';
import '../services/image_picker_service.dart';
import '../theme/app_colors.dart';
import '../widgets/app_card.dart';
import '../widgets/app_button.dart';
import '../widgets/upload_row.dart';
import '../widgets/health_image_slot.dart';
import '../widgets/output_panel.dart';
import '../widgets/capture_guidance_banner.dart';
import '../widgets/results/prediction_result_view.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  Map<String, dynamic>? _health;
  Map<String, dynamic>? _prediction;
  bool _loading = false;
  String? _error;

  PickedImage? _healthTop;
  PickedImage? _healthBottom;

  Future<void> _checkHealth() async {
    setState(() => _error = null);
    try {
      final result = await ApiService.checkHealth();
      setState(() => _health = result);
    } on ApiException catch (e) {
      setState(() => _error = e.message);
    }
  }

  Future<void> _handleUpload(String endpoint, ImageSource source) async {
    final picked = await ImagePickerService.pick(source);
    if (picked == null) return;

    setState(() {
      _loading = true;
      _error = null;
      _prediction = null;
    });

    try {
      final result = await ApiService.predict(endpoint: endpoint, image: picked);
      setState(() => _prediction = result);
    } on ApiException catch (e) {
      setState(() => _error = e.message);
    } finally {
      setState(() => _loading = false);
    }
  }

  Future<void> _pickHealthTop(ImageSource source) async {
    final img = await ImagePickerService.pick(source);
    if (img == null) return;
    setState(() {
      _healthTop = img;
      _error = null;
    });
  }

  Future<void> _pickHealthBottom(ImageSource source) async {
    final img = await ImagePickerService.pick(source);
    if (img == null) return;
    setState(() {
      _healthBottom = img;
      _error = null;
    });
  }

  Future<void> _submitLeafHealth() async {
    if (_healthTop == null || _healthBottom == null) {
      setState(() => _error = 'Please select both a top and a bottom leaf image first.');
      return;
    }

    setState(() {
      _loading = true;
      _error = null;
      _prediction = null;
    });

    try {
      final result =
          await ApiService.predictLeafHealth(top: _healthTop!, bottom: _healthBottom!);
      setState(() {
        _prediction = result;
        _healthTop = null;
        _healthBottom = null;
      });
    } on ApiException catch (e) {
      setState(() => _error = e.message);
    } finally {
      setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [AppColors.bgStart, AppColors.bgEnd],
          ),
        ),
        child: SafeArea(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(32),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                const Text(
                  '🌿 Ayurveda Plant Recognition',
                  textAlign: TextAlign.center,
                  style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold, color: AppColors.titleColor),
                ),
                const SizedBox(height: 32),

                AppCard(
                  title: 'API Health',
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      AppButton(label: 'Ping Backend', onPressed: _checkHealth),
                      if (_health != null) ...[
                        const SizedBox(height: 12),
                        OutputPanel(text: const JsonEncoder.withIndent('  ').convert(_health)),
                      ],
                    ],
                  ),
                ),
                const SizedBox(height: 24),

                AppCard(
                  title: 'Test Prediction',
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      UploadRow(
                        emoji: '🌸',
                        label: 'Flower Image',
                        guidance: CaptureSubject.flower,
                        onPick: (source) => _handleUpload('flower', source),
                      ),
                      const SizedBox(height: 16),
                      UploadRow(
                        emoji: '🍃',
                        label: 'Leaf Image (auto-detects simple or compound)',
                        guidance: CaptureSubject.leaf,
                        onPick: (source) => _handleUpload('leaf', source),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 24),

                AppCard(
                  title: 'Leaf Health Assessment',
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      const Text(
                        'Upload both the top and bottom of the same leaf to assess its health.',
                        style: TextStyle(color: AppColors.textPrimary, fontSize: 13),
                      ),
                      const SizedBox(height: 12),
                      HealthImageSlot(
                        emoji: '⬆️',
                        label: 'Top of Leaf',
                        picked: _healthTop,
                        guidance: CaptureSubject.leaf,
                        onPick: _pickHealthTop,
                      ),
                      const SizedBox(height: 12),
                      HealthImageSlot(
                        emoji: '⬇️',
                        label: 'Bottom of Leaf',
                        picked: _healthBottom,
                        onPick: _pickHealthBottom,
                      ),
                      const SizedBox(height: 16),
                      AppButton(label: '🩺 Assess Leaf Health', onPressed: _submitLeafHealth),
                    ],
                  ),
                ),
                const SizedBox(height: 16),

                if (_loading)
                  const Text(
                    'Analyzing plant...',
                    style: TextStyle(color: AppColors.loadingColor, fontWeight: FontWeight.bold, fontSize: 15),
                  ),

                if (_error != null)
                  Text(
                    _error!,
                    style: const TextStyle(color: AppColors.errorColor, fontWeight: FontWeight.bold, fontSize: 15),
                  ),

                if (_prediction != null) ...[
                  const SizedBox(height: 8),
                  AppCard(title: 'Prediction Result', child: PredictionResultView(data: _prediction!)),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }
}