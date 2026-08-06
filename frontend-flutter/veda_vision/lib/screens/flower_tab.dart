import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../services/api_service.dart';
import '../services/image_picker_service.dart';
import '../theme/app_colors.dart';
import '../widgets/app_card.dart';
import '../widgets/upload_row.dart';
import '../widgets/capture_guidance_banner.dart';
import '../widgets/result_sheet.dart';

class FlowerTab extends StatefulWidget {
  const FlowerTab({super.key});

  @override
  State<FlowerTab> createState() => _FlowerTabState();
}

class _FlowerTabState extends State<FlowerTab> {
  bool _loading = false;
  String? _error;

  Future<void> _handleUpload(ImageSource source) async {
    final picked = await ImagePickerService.pick(source);
    if (picked == null) return;

    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final result = await ApiService.predict(endpoint: 'flower', image: picked);
      if (!mounted) return;
      await showResultSheet(context, result);
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
            title: 'Identify a Flower',
            child: UploadRow(
              emoji: '🌸',
              label: 'Flower Image',
              guidance: CaptureSubject.flower,
              onPick: _handleUpload,
            ),
          ),
          const SizedBox(height: 16),
          if (_loading)
            const Padding(
              padding: EdgeInsets.only(top: 8),
              child: Text(
                'Analyzing plant...',
                style: TextStyle(
                  color: AppColors.loadingColor,
                  fontWeight: FontWeight.bold,
                  fontSize: 15,
                ),
              ),
            ),
          if (_error != null)
            Padding(
              padding: const EdgeInsets.only(top: 8),
              child: Text(
                _error!,
                style: const TextStyle(
                  color: AppColors.errorColor,
                  fontWeight: FontWeight.bold,
                  fontSize: 15,
                ),
              ),
            ),
        ],
      ),
    );
  }
}