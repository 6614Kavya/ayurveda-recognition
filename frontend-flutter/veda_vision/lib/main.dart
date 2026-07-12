import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'dart:io';

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────
void main() {
  runApp(const AyurvedaApp());
}

// ─────────────────────────────────────────────────────────────────────────────
// Theme colors — mirrors the React style constants exactly
// ─────────────────────────────────────────────────────────────────────────────
class AppColors {
  // background gradient: #f5f1e6 → #e8f5e9
  static const bgStart      = Color(0xFFF5F1E6);
  static const bgEnd        = Color(0xFFE8F5E9);

  // text
  static const textPrimary  = Color(0xFF2E4D34);   // #2e4d34
  static const titleColor   = Color(0xFF355E3B);   // #355e3b
  static const sectionTitle = Color(0xFF4A7C59);   // #4a7c59

  // card
  static const cardBg       = Color(0xCCFFFFFF);   // #ffffffcc
  static const cardShadow   = Color(0x14000000);   // rgba(0,0,0,0.08)

  // button
  static const buttonBg     = Color(0xFF6B8E23);   // #6b8e23
  static const buttonText   = Colors.white;

  // output panel
  static const outputBg     = Color(0xFFF0F7F2);   // #f0f7f2

  // input border
  static const inputBorder  = Color(0xFFCCCCCC);   // #ccc

  // loading / error
  static const loadingColor = Color(0xFF6B8E23);
  static const errorColor   = Color(0xFFC0392B);   // #c0392b
}

// ─────────────────────────────────────────────────────────────────────────────
// Root app
// ─────────────────────────────────────────────────────────────────────────────
class AyurvedaApp extends StatelessWidget {
  const AyurvedaApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Ayurveda Plant Recognition',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        fontFamily: 'Segoe UI',
        colorScheme: ColorScheme.fromSeed(seedColor: AppColors.titleColor),
        useMaterial3: true,
      ),
      home: const HomePage(),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Home page — mirrors App() in React
// ─────────────────────────────────────────────────────────────────────────────
class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  static const String apiBase = 'http://localhost:8000';

  Map<String, dynamic>? _health;
  Map<String, dynamic>? _prediction;
  bool _loading  = false;
  String? _error;

  // ── API: ping health endpoint ───────────────────────────────────
  Future<void> _checkHealth() async {
    setState(() { _error = null; });
    try {
      final res = await http.get(Uri.parse('$apiBase/health'));
      setState(() {
        _health = jsonDecode(res.body) as Map<String, dynamic>;
      });
    } catch (_) {
      setState(() {
        _error = 'Could not reach API — is uvicorn running?';
      });
    }
  }

  // ── API: upload image to a prediction endpoint ──────────────────
  // endpoint: 'flower' | 'single-leaf' | 'compound-leaf'
  Future<void> _handleUpload(String endpoint) async {
    final picker = ImagePicker();
    final picked = await picker.pickImage(source: ImageSource.gallery);
    if (picked == null) return;

    setState(() {
      _loading    = true;
      _error      = null;
      _prediction = null;
    });

    try {
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$apiBase/predict/$endpoint'),
      );
      request.files.add(
        await http.MultipartFile.fromPath('file', picked.path),
      );
      final streamed = await request.send();
      final res      = await http.Response.fromStream(streamed);
      final body     = jsonDecode(res.body);

      if (res.statusCode == 200) {
        setState(() { _prediction = body as Map<String, dynamic>; });
      } else {
        setState(() {
          _error = (body as Map<String, dynamic>)['detail']?.toString()
                   ?? 'Upload failed';
        });
      }
    } catch (e) {
      setState(() { _error = 'Upload failed: $e'; });
    } finally {
      setState(() { _loading = false; });
    }
  }

  // ── Build ───────────────────────────────────────────────────────
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        // Gradient background — mirrors React's linear-gradient(135deg, ...)
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end:   Alignment.bottomRight,
            colors: [AppColors.bgStart, AppColors.bgEnd],
          ),
        ),
        child: SafeArea(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(32),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // ── Title ──────────────────────────────────────────
                const Text(
                  '🌿 Ayurveda Plant Recognition',
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 28,
                    fontWeight: FontWeight.bold,
                    color: AppColors.titleColor,
                  ),
                ),
                const SizedBox(height: 32),

                // ── Health Card ────────────────────────────────────
                _AppCard(
                  title: 'API Health',
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _AppButton(
                        label: 'Ping Backend',
                        onPressed: _checkHealth,
                      ),
                      if (_health != null) ...[
                        const SizedBox(height: 12),
                        _OutputPanel(
                          text: const JsonEncoder.withIndent('  ')
                              .convert(_health),
                        ),
                      ],
                    ],
                  ),
                ),
                const SizedBox(height: 24),

                // ── Prediction Card ────────────────────────────────
                _AppCard(
                  title: 'Test Prediction',
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      _UploadRow(
                        emoji:    '🌸',
                        label:    'Flower Image',
                        onPick:   () => _handleUpload('flower'),
                      ),
                      const SizedBox(height: 16),
                      _UploadRow(
                        emoji:    '🍃',
                        label:    'Single Leaf Image',
                        onPick:   () => _handleUpload('single-leaf'),
                      ),
                      const SizedBox(height: 16),
                      _UploadRow(
                        emoji:    '🌿',
                        label:    'Compound Leaf Image',
                        onPick:   () => _handleUpload('compound-leaf'),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 16),

                // ── Loading / Error states ─────────────────────────
                if (_loading)
                  const Text(
                    'Analyzing plant...',
                    style: TextStyle(
                      color:      AppColors.loadingColor,
                      fontWeight: FontWeight.bold,
                      fontSize:   15,
                    ),
                  ),

                if (_error != null)
                  Text(
                    _error!,
                    style: const TextStyle(
                      color:      AppColors.errorColor,
                      fontWeight: FontWeight.bold,
                      fontSize:   15,
                    ),
                  ),

                // ── Prediction Result Card ─────────────────────────
                if (_prediction != null) ...[
                  const SizedBox(height: 8),
                  _AppCard(
                    title: 'Prediction Result',
                    child: _OutputPanel(
                      text: const JsonEncoder.withIndent('  ')
                          .convert(_prediction),
                    ),
                  ),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Reusable card — mirrors .card style in React
// ─────────────────────────────────────────────────────────────────────────────
class _AppCard extends StatelessWidget {
  final String title;
  final Widget child;

  const _AppCard({required this.title, required this.child});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color:        AppColors.cardBg,
        borderRadius: BorderRadius.circular(16),
        boxShadow: const [
          BoxShadow(
            color:      AppColors.cardShadow,
            blurRadius: 20,
            offset:     Offset(0, 8),
          ),
        ],
      ),
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: const TextStyle(
              fontSize:   18,
              fontWeight: FontWeight.w600,
              color:      AppColors.sectionTitle,
            ),
          ),
          const SizedBox(height: 16),
          child,
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Button — mirrors .button style in React
// ─────────────────────────────────────────────────────────────────────────────
class _AppButton extends StatelessWidget {
  final String label;
  final VoidCallback onPressed;

  const _AppButton({required this.label, required this.onPressed});

  @override
  Widget build(BuildContext context) {
    return ElevatedButton(
      onPressed: onPressed,
      style: ElevatedButton.styleFrom(
        backgroundColor: AppColors.buttonBg,
        foregroundColor: AppColors.buttonText,
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(8),
        ),
        elevation: 0,
      ),
      child: Text(
        label,
        style: const TextStyle(fontSize: 14),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Upload row — mirrors <label> + <input type="file"> in React
// Each row shows a label and a "Choose File" button that opens gallery
// ─────────────────────────────────────────────────────────────────────────────
class _UploadRow extends StatelessWidget {
  final String    emoji;
  final String    label;
  final VoidCallback onPick;

  const _UploadRow({
    required this.emoji,
    required this.label,
    required this.onPick,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        border:       Border.all(color: AppColors.inputBorder),
        borderRadius: BorderRadius.circular(6),
      ),
      child: Row(
        children: [
          // Label text
          Expanded(
            child: Text(
              '$emoji $label',
              style: const TextStyle(
                fontWeight: FontWeight.w500,
                color:      AppColors.textPrimary,
                fontSize:   14,
              ),
            ),
          ),
          // File picker button
          _AppButton(label: 'Choose File', onPressed: onPick),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Output panel — mirrors .output / <pre> in React
// ─────────────────────────────────────────────────────────────────────────────
class _OutputPanel extends StatelessWidget {
  final String text;

  const _OutputPanel({required this.text});

  @override
  Widget build(BuildContext context) {
    return Container(
      width:       double.infinity,
      padding:     const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color:        AppColors.outputBg,
        borderRadius: BorderRadius.circular(10),
      ),
      child: Text(
        text,
        style: const TextStyle(
          fontFamily: 'monospace',
          fontSize:   13,
          color:      AppColors.textPrimary,
          height:     1.5,
        ),
      ),
    );
  }
}
