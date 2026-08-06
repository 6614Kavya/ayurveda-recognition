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
import '../widgets/capture_guidance_banner.dart';
import '../widgets/result_sheet.dart';

class LeafTab extends StatefulWidget {
  const LeafTab({super.key});

  @override
  State<LeafTab> createState() => _LeafTabState();
}

class _LeafTabState extends State<LeafTab> {
  bool _loading = false;
  String? _error;

  PickedImage? _healthTop;
  PickedImage? _healthBottom;

  // ── Leaf identification ──────────────────────────────────────────
  Future<void> _handleIdentify(ImageSource source) async {
    final picked = await ImagePickerService.pick(source);
    if (picked == null) return;

    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final result = await ApiService.predict(endpoint: 'leaf', image: picked);
      if (!mounted) return;
      await showResultSheet(context, result);
    } on ApiException catch (e) {
      setState(() => _error = e.message);
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  // ── Leaf health assessment ───────────────────────────────────────
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
    });

    try {
      final result =
          await ApiService.predictLeafHealth(top: _healthTop!, bottom: _healthBottom!);
      if (!mounted) return;
      await showResultSheet(context, result);
      setState(() {
        _healthTop = null;
        _healthBottom = null;
      });
    } on ApiException catch (e) {
      setState(() => _error = e.message);
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          AppCard(
            title: 'Identify a Leaf',
            child: UploadRow(
              emoji: '🍃',
              label: 'Leaf Image (auto-detects simple or compound)',
              guidance: CaptureSubject.leaf,
              onPick: _handleIdentify,
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
                const SizedBox(height: 10),
                const CaptureGuidanceBanner(subject: CaptureSubject.leaf),
                const SizedBox(height: 4),
                HealthImageSlot(
                  emoji: '⬆️',
                  label: 'Top of Leaf',
                  picked: _healthTop,
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
              style: TextStyle(
                color: AppColors.loadingColor,
                fontWeight: FontWeight.bold,
                fontSize: 15,
              ),
            ),
          if (_error != null)
            Text(
              _error!,
              style: const TextStyle(
                color: AppColors.errorColor,
                fontWeight: FontWeight.bold,
                fontSize: 15,
              ),
            ),
        ],
      ),
    );
  }
}