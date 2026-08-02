import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:http_parser/http_parser.dart';
import 'dart:io' show Platform;
import 'package:flutter/foundation.dart' show kIsWeb, Uint8List;
import 'package:file_selector/file_selector.dart' as fs;

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
  static const bgStart = Color(0xFFF5F1E6);
  static const bgEnd = Color(0xFFE8F5E9);

  // text
  static const textPrimary = Color(0xFF2E4D34); // #2e4d34
  static const titleColor = Color(0xFF355E3B); // #355e3b
  static const sectionTitle = Color(0xFF4A7C59); // #4a7c59

  // card
  static const cardBg = Color(0xCCFFFFFF); // #ffffffcc
  static const cardShadow = Color(0x14000000); // rgba(0,0,0,0.08)

  // button
  static const buttonBg = Color(0xFF6B8E23); // #6b8e23
  static const buttonText = Colors.white;

  // output panel
  static const outputBg = Color(0xFFF0F7F2); // #f0f7f2

  // input border
  static const inputBorder = Color(0xFFCCCCCC); // #ccc

  // loading / error
  static const loadingColor = Color(0xFF6B8E23);
  static const errorColor = Color(0xFFC0392B); // #c0392b

  // amber (used for mid-range confidence / decision states)
  static const amber = Color(0xFFD98C2B);
}

// ─────────────────────────────────────────────────────────────────────────────
// Small shared helper — normalizes a confidence value to a 0..1 fraction,
// whether the backend sent it as 0..1 or 0..100.
// ─────────────────────────────────────────────────────────────────────────────
double _asFraction(dynamic v) {
  final d = (v is num) ? v.toDouble() : double.tryParse(v?.toString() ?? '') ?? 0.0;
  return d > 1.0 ? d / 100.0 : d;
}

double _parsePercent(dynamic v) {
  if (v == null) return 0.0;
  if (v is num) return v > 1.0 ? v / 100.0 : v.toDouble();
  final s = v.toString().trim();
  if (s.endsWith('%')) {
    final n = double.tryParse(s.substring(0, s.length - 1)) ?? 0.0;
    return n / 100.0;
  }
  final n = double.tryParse(s) ?? 0.0;
  return n > 1.0 ? n / 100.0 : n;
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
// Small holder for a picked image's bytes + filename, used by the leaf-health
// flow (which needs to hold two images in state before submitting them
// together).
// ─────────────────────────────────────────────────────────────────────────────
class _PickedImage {
  final Uint8List bytes;
  final String filename;
  const _PickedImage({required this.bytes, required this.filename});
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
  bool _loading = false;
  String? _error;

  // Leaf-health flow needs two images held in state before submission.
  _PickedImage? _healthTop;
  _PickedImage? _healthBottom;

  // ── API: ping health endpoint ───────────────────────────────────
  Future<void> _checkHealth() async {
    setState(() {
      _error = null;
    });
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

  // ── API: upload image to a single-file prediction endpoint ──────
  // endpoint: 'flower' (unchanged) | 'leaf' (auto-routes simple/compound
  // server-side via /predict/leaf).
  //
  // NOT MODIFIED for the flower path — same behavior as before.
  Future<void> _handleUpload(String endpoint, ImageSource source) async {
    Uint8List bytes;
    String filename;

    if (_isDesktop && source == ImageSource.gallery) {
      // Desktop gallery pick: use file_selector directly with a widened
      // extension filter so HEIC/HEIF show up (image_picker's Windows/Linux/
      // macOS path hardcodes jpg/jpeg/png/bmp/webp/gif and excludes heic).
      final typeGroup = fs.XTypeGroup(
        label: 'images',
        extensions: ['jpg', 'jpeg', 'png', 'bmp', 'webp', 'heic', 'heif'],
      );
      final file = await fs.openFile(acceptedTypeGroups: [typeGroup]);
      if (file == null) return; // user cancelled

      bytes = await file.readAsBytes();
      filename = file.name;
    } else {
      // Mobile (camera + gallery) and desktop camera: image_picker handles
      // these fine, including HEIC on iOS/Android.
      final picker = ImagePicker();
      final picked = await picker.pickImage(source: source);
      if (picked == null) return; // user cancelled

      bytes = await picked.readAsBytes();
      filename = picked.name;
    }

    setState(() {
      _loading = true;
      _error = null;
      _prediction = null;
    });

    try {
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$apiBase/predict/$endpoint'),
      );

      request.files.add(
        http.MultipartFile.fromBytes(
          'file',
          bytes,
          filename: filename,
          contentType: MediaType.parse(_mimeTypeFor(filename)),
        ),
      );

      final streamed = await request.send();
      final res = await http.Response.fromStream(streamed);
      final body = jsonDecode(res.body);

      if (res.statusCode == 200) {
        setState(() => _prediction = body as Map<String, dynamic>);
      } else {
        setState(() {
          _error =
              (body as Map<String, dynamic>)['detail']?.toString() ??
              'Upload failed';
        });
      }
    } catch (e) {
      setState(() => _error = 'Upload failed: $e');
    } finally {
      setState(() => _loading = false);
    }
  }

  // ── Shared picker used only by the leaf-health flow ──────────────
  // Kept separate from _handleUpload above so the flower/leaf identification
  // path is never touched by this change.
  Future<_PickedImage?> _pickImageBytes(ImageSource source) async {
    if (_isDesktop && source == ImageSource.gallery) {
      final typeGroup = fs.XTypeGroup(
        label: 'images',
        extensions: ['jpg', 'jpeg', 'png', 'bmp', 'webp', 'heic', 'heif'],
      );
      final file = await fs.openFile(acceptedTypeGroups: [typeGroup]);
      if (file == null) return null;
      final bytes = await file.readAsBytes();
      return _PickedImage(bytes: bytes, filename: file.name);
    } else {
      final picker = ImagePicker();
      final picked = await picker.pickImage(source: source);
      if (picked == null) return null;
      final bytes = await picked.readAsBytes();
      return _PickedImage(bytes: bytes, filename: picked.name);
    }
  }

  Future<void> _pickHealthTop(ImageSource source) async {
    final img = await _pickImageBytes(source);
    if (img == null) return;
    setState(() {
      _healthTop = img;
      _error = null;
    });
  }

  Future<void> _pickHealthBottom(ImageSource source) async {
    final img = await _pickImageBytes(source);
    if (img == null) return;
    setState(() {
      _healthBottom = img;
      _error = null;
    });
  }

  // ── API: submit both top + bottom images to /predict/leaf-health ─
  Future<void> _submitLeafHealth() async {
    if (_healthTop == null || _healthBottom == null) {
      setState(() {
        _error = 'Please select both a top and a bottom leaf image first.';
      });
      return;
    }

    setState(() {
      _loading = true;
      _error = null;
      _prediction = null;
    });

    try {
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$apiBase/predict/leaf-health'),
      );

      request.files.add(
        http.MultipartFile.fromBytes(
          'top_file',
          _healthTop!.bytes,
          filename: _healthTop!.filename,
          contentType: MediaType.parse(_mimeTypeFor(_healthTop!.filename)),
        ),
      );
      request.files.add(
        http.MultipartFile.fromBytes(
          'bottom_file',
          _healthBottom!.bytes,
          filename: _healthBottom!.filename,
          contentType: MediaType.parse(_mimeTypeFor(_healthBottom!.filename)),
        ),
      );

      final streamed = await request.send();
      final res = await http.Response.fromStream(streamed);
      final body = jsonDecode(res.body);

      if (res.statusCode == 200) {
        setState(() {
          _prediction = body as Map<String, dynamic>;
          _healthTop = null;
          _healthBottom = null;
        });
      } else {
        setState(() {
          _error =
              (body as Map<String, dynamic>)['detail']?.toString() ??
              'Upload failed';
        });
      }
    } catch (e) {
      setState(() => _error = 'Upload failed: $e');
    } finally {
      setState(() => _loading = false);
    }
  }

  // Detects desktop platforms (Windows/Linux/macOS) — this is where
  // image_picker's gallery filter can't be widened, so we route to
  // file_selector instead. Web and mobile keep using image_picker.
  bool get _isDesktop =>
      !kIsWeb && (Platform.isWindows || Platform.isLinux || Platform.isMacOS);

  // Proper extension → MIME type mapping, including HEIC/HEIF
  String _mimeTypeFor(String filename) {
    final ext = filename.split('.').last.toLowerCase();
    switch (ext) {
      case 'png':
        return 'image/png';
      case 'heic':
        return 'image/heic';
      case 'heif':
        return 'image/heif';
      case 'webp':
        return 'image/webp';
      case 'jpg':
      case 'jpeg':
      default:
        return 'image/jpeg';
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
                          text: const JsonEncoder.withIndent(
                            '  ',
                          ).convert(_health),
                        ),
                      ],
                    ],
                  ),
                ),
                const SizedBox(height: 24),

                // ── Prediction Card (flower + leaf identification) ─
                _AppCard(
                  title: 'Test Prediction',
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      // Flower flow — endpoint and handler unchanged.
                      _UploadRow(
                        emoji: '🌸',
                        label: 'Flower Image',
                        onPick: (source) => _handleUpload('flower', source),
                      ),
                      const SizedBox(height: 16),
                      // Single-leaf / compound-leaf rows replaced with one
                      // row hitting /predict/leaf, which now decides
                      // simple-vs-compound and dispatches internally.
                      _UploadRow(
                        emoji: '🍃',
                        label: 'Leaf Image (auto-detects simple or compound)',
                        onPick: (source) => _handleUpload('leaf', source),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 24),

                // ── Leaf Health Card (needs top + bottom images) ────
                _AppCard(
                  title: 'Leaf Health Assessment',
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      const Text(
                        'Upload both the top and bottom of the same leaf to assess its health.',
                        style: TextStyle(
                          color: AppColors.textPrimary,
                          fontSize: 13,
                        ),
                      ),
                      const SizedBox(height: 12),
                      _HealthImageSlot(
                        emoji: '⬆️',
                        label: 'Top of Leaf',
                        picked: _healthTop,
                        onPick: _pickHealthTop,
                      ),
                      const SizedBox(height: 12),
                      _HealthImageSlot(
                        emoji: '⬇️',
                        label: 'Bottom of Leaf',
                        picked: _healthBottom,
                        onPick: _pickHealthBottom,
                      ),
                      const SizedBox(height: 16),
                      _AppButton(
                        label: '🩺 Assess Leaf Health',
                        onPressed: _submitLeafHealth,
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

                // ── Prediction Result Card ─────────────────────────
                if (_prediction != null) ...[
                  const SizedBox(height: 8),
                  _AppCard(
                    title: 'Prediction Result',
                    child: _PredictionResultView(data: _prediction!),
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
        color: AppColors.cardBg,
        borderRadius: BorderRadius.circular(16),
        boxShadow: const [
          BoxShadow(
            color: AppColors.cardShadow,
            blurRadius: 20,
            offset: Offset(0, 8),
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
              fontSize: 18,
              fontWeight: FontWeight.w600,
              color: AppColors.sectionTitle,
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
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
        elevation: 0,
      ),
      child: Text(label, style: const TextStyle(fontSize: 14)),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Upload row — mirrors <label> + <input type="file"> in React
// Each row shows a label and Camera/Gallery buttons; used by flower + leaf.
// ─────────────────────────────────────────────────────────────────────────────
class _UploadRow extends StatelessWidget {
  final String emoji;
  final String label;
  final void Function(ImageSource) onPick;

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
        border: Border.all(color: AppColors.inputBorder),
        borderRadius: BorderRadius.circular(6),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            '$emoji $label',
            style: const TextStyle(
              fontWeight: FontWeight.w500,
              color: AppColors.textPrimary,
              fontSize: 14,
            ),
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              _AppButton(
                label: '📷 Camera',
                onPressed: () => onPick(ImageSource.camera),
              ),
              const SizedBox(width: 8),
              _AppButton(
                label: '🖼 Gallery',
                onPressed: () => onPick(ImageSource.gallery),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Health image slot — like _UploadRow, but holds onto the picked image
// (thumbnail + filename + checkmark) instead of uploading immediately,
// since leaf-health needs two images submitted together.
// ─────────────────────────────────────────────────────────────────────────────
class _HealthImageSlot extends StatelessWidget {
  final String emoji;
  final String label;
  final _PickedImage? picked;
  final void Function(ImageSource) onPick;

  const _HealthImageSlot({
    required this.emoji,
    required this.label,
    required this.picked,
    required this.onPick,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        border: Border.all(
          color: picked != null ? AppColors.buttonBg : AppColors.inputBorder,
          width: picked != null ? 1.5 : 1,
        ),
        borderRadius: BorderRadius.circular(6),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (picked != null)
            ClipRRect(
              borderRadius: BorderRadius.circular(6),
              child: Image.memory(
                picked!.bytes,
                width: 48,
                height: 48,
                fit: BoxFit.cover,
              ),
            )
          else
            Container(
              width: 48,
              height: 48,
              decoration: BoxDecoration(
                color: AppColors.outputBg,
                borderRadius: BorderRadius.circular(6),
              ),
              alignment: Alignment.center,
              child: Text(emoji, style: const TextStyle(fontSize: 20)),
            ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Text(
                      '$emoji $label',
                      style: const TextStyle(
                        fontWeight: FontWeight.w500,
                        color: AppColors.textPrimary,
                        fontSize: 14,
                      ),
                    ),
                    if (picked != null) ...[
                      const SizedBox(width: 6),
                      const Icon(
                        Icons.check_circle,
                        color: AppColors.buttonBg,
                        size: 16,
                      ),
                    ],
                  ],
                ),
                if (picked != null)
                  Padding(
                    padding: const EdgeInsets.only(top: 2),
                    child: Text(
                      picked!.filename,
                      style: const TextStyle(
                        fontSize: 11,
                        color: Colors.black54,
                      ),
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    _AppButton(
                      label: '📷 Camera',
                      onPressed: () => onPick(ImageSource.camera),
                    ),
                    const SizedBox(width: 8),
                    _AppButton(
                      label: '🖼 Gallery',
                      onPressed: () => onPick(ImageSource.gallery),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Output panel — mirrors .output / <pre> in React. Now only used for the
// /health ping and as a fallback for unrecognized prediction response shapes.
// ─────────────────────────────────────────────────────────────────────────────
class _OutputPanel extends StatelessWidget {
  final String text;

  const _OutputPanel({required this.text});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.outputBg,
        borderRadius: BorderRadius.circular(10),
      ),
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

// ─────────────────────────────────────────────────────────────────────────────
// Prediction result dispatcher — decides which nicely-formatted view to show
// based on the shape of the JSON response, instead of always dumping raw
// JSON. Falls back to raw JSON only if the shape isn't recognized, so
// nothing is ever silently hidden.
// ─────────────────────────────────────────────────────────────────────────────
class _PredictionResultView extends StatelessWidget {
  final Map<String, dynamic> data;
  const _PredictionResultView({required this.data});

  @override
  Widget build(BuildContext context) {
    if (data.containsKey('health_result')) {
      return _LeafHealthResultView(data: data);
    }
    if (data.containsKey('plant_name')) {
      return _IdentificationResultView(data: data);
    }
    return _OutputPanel(text: const JsonEncoder.withIndent('  ').convert(data));
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Small confidence meter — label + percentage + color-coded progress bar.
// Green ≥80%, amber 50–79%, red <50%.
// ─────────────────────────────────────────────────────────────────────────────
class _ConfidenceMeter extends StatelessWidget {
  final String label;
  final double fraction; // 0..1

  const _ConfidenceMeter({required this.label, required this.fraction});

  Color get _color {
    if (fraction >= 0.8) return AppColors.buttonBg;
    if (fraction >= 0.5) return AppColors.amber;
    return AppColors.errorColor;
  }

  @override
  Widget build(BuildContext context) {
    final pct = (fraction.clamp(0.0, 1.0) * 100).toStringAsFixed(1);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              label,
              style: const TextStyle(
                fontSize: 12,
                color: AppColors.textPrimary,
                fontWeight: FontWeight.w500,
              ),
            ),
            Text(
              '$pct%',
              style: TextStyle(
                fontSize: 12,
                color: _color,
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
        const SizedBox(height: 4),
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: LinearProgressIndicator(
            value: fraction.clamp(0.0, 1.0),
            minHeight: 8,
            backgroundColor: AppColors.inputBorder.withOpacity(0.4),
            valueColor: AlwaysStoppedAnimation<Color>(_color),
          ),
        ),
      ],
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Small pill badge — used for module names, species names, decisions, and
// disease chips.
// ─────────────────────────────────────────────────────────────────────────────
class _Badge extends StatelessWidget {
  final String text;
  final Color color;
  const _Badge({required this.text, required this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: color.withOpacity(0.15),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color.withOpacity(0.4)),
      ),
      child: Text(
        text,
        style: TextStyle(color: color, fontSize: 12, fontWeight: FontWeight.w600),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// A single bullet line, used for the "Traditional Uses" list.
// ─────────────────────────────────────────────────────────────────────────────
class _BulletLine extends StatelessWidget {
  final String text;
  const _BulletLine({required this.text});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            '•  ',
            style: TextStyle(color: AppColors.buttonBg, fontWeight: FontWeight.bold),
          ),
          Expanded(
            child: Text(
              text,
              style: const TextStyle(
                fontSize: 13,
                color: AppColors.textPrimary,
                height: 1.4,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// A small labeled stat box — used for Health Value / Severity Score.
// ─────────────────────────────────────────────────────────────────────────────
class _StatTile extends StatelessWidget {
  final String label;
  final String value;
  const _StatTile({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: AppColors.outputBg,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label, style: const TextStyle(fontSize: 11, color: Colors.black54)),
          const SizedBox(height: 4),
          Text(
            value,
            style: const TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.bold,
              color: AppColors.titleColor,
            ),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// A single label/value row, used inside the health "breakdown" table.
// ─────────────────────────────────────────────────────────────────────────────
class _KeyValueRow extends StatelessWidget {
  final String label;
  final String value;
  const _KeyValueRow({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Expanded(
            child: Text(label, style: const TextStyle(fontSize: 12, color: AppColors.textPrimary)),
          ),
          Text(
            value,
            style: const TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: AppColors.textPrimary,
            ),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Formatted view for plant identification results (flower or leaf), i.e.
// responses shaped like PredictionResponse: plant_name, confidence, module,
// sinhala_name, uses, diseases_treated.
// ─────────────────────────────────────────────────────────────────────────────
class _IdentificationResultView extends StatelessWidget {
  final Map<String, dynamic> data;
  const _IdentificationResultView({required this.data});

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
    final confidence = _asFraction(data['confidence']);
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
              child: Text(
                plantName,
                style: const TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: AppColors.titleColor,
                ),
              ),
            ),
          ],
        ),
        if (sinhalaName != null && sinhalaName.isNotEmpty)
          Padding(
            padding: const EdgeInsets.only(top: 2, left: 26),
            child: Text(
              sinhalaName,
              style: const TextStyle(
                fontStyle: FontStyle.italic,
                color: Colors.black54,
                fontSize: 13,
              ),
            ),
          ),
        const SizedBox(height: 12),
        if (module != null) _Badge(text: _friendlyModule(module), color: AppColors.sectionTitle),
        const SizedBox(height: 12),
        _ConfidenceMeter(label: 'Prediction confidence', fraction: confidence),
        if (uses.isNotEmpty) ...[
          const SizedBox(height: 16),
          const Text(
            'Traditional Uses',
            style: TextStyle(
              fontWeight: FontWeight.w600,
              color: AppColors.sectionTitle,
              fontSize: 13,
            ),
          ),
          const SizedBox(height: 6),
          ...uses.map((u) => _BulletLine(text: u)),
        ],
        if (diseases.isNotEmpty) ...[
          const SizedBox(height: 16),
          const Text(
            'Diseases Treated',
            style: TextStyle(
              fontWeight: FontWeight.w600,
              color: AppColors.sectionTitle,
              fontSize: 13,
            ),
          ),
          const SizedBox(height: 6),
          Wrap(
            spacing: 6,
            runSpacing: 6,
            children: diseases.map((d) => _Badge(text: d, color: AppColors.buttonBg)).toList(),
          ),
        ],
      ],
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Formatted view for /predict/leaf-health responses: leaf_type, confidence,
// health_result { species, decision, decision_confidence, health_value,
// severity_score_raw, breakdown }.
// ─────────────────────────────────────────────────────────────────────────────
class _LeafHealthResultView extends StatelessWidget {
  final Map<String, dynamic> data;
  const _LeafHealthResultView({required this.data});

  String _titleCase(String s) => s
      .split('_')
      .map((w) => w.isEmpty ? w : '${w[0].toUpperCase()}${w.substring(1)}')
      .join(' ');

  Color _decisionColor(String decision) {
    final d = decision.toLowerCase();
    if (d.contains('healthy') || d.contains('good')) return AppColors.buttonBg;
    if (d.contains('unhealthy') ||
        d.contains('disease') ||
        d.contains('infect') ||
        d.contains('poor')) {
      return AppColors.errorColor;
    }
    return AppColors.amber;
  }

  String _formatValue(dynamic v) {
    if (v is int) return v.toString();
    if (v is double) return v.toStringAsFixed(2);
    return v?.toString() ?? '—';
  }

  @override
  Widget build(BuildContext context) {
    final leafType = data['leaf_type']?.toString() ?? '';
    final routeConfidence = _asFraction(data['confidence']);
    final health = (data['health_result'] as Map?)?.cast<String, dynamic>() ?? {};

    // Simple-leaf response uses stage1_status / stage1_confidence / health_percentage.
    final isSimpleStage = health.containsKey('stage1_status');

    if (isSimpleStage) {
      final decision = health['stage1_status']?.toString() ?? 'Unknown';
      final decisionConfidence = _parsePercent(health['stage1_confidence']);
      final healthPercentage = health['health_percentage']?.toString() ?? '—';

      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              _Badge(
                text: leafType.isEmpty
                    ? 'Leaf'
                    : '${leafType[0].toUpperCase()}${leafType.substring(1)} Leaf',
                color: AppColors.sectionTitle,
              ),
              _Badge(text: decision, color: _decisionColor(decision)),
            ],
          ),
          const SizedBox(height: 14),
          _ConfidenceMeter(label: 'Leaf-type routing confidence', fraction: routeConfidence),
          const SizedBox(height: 10),
          _ConfidenceMeter(label: 'Stage 1 confidence', fraction: decisionConfidence),
          const SizedBox(height: 16),
          _StatTile(label: 'Health Percentage', value: healthPercentage),
        ],
      );
    }

    // ── Compound-leaf (unchanged) shape ────────────────────────────────
    final species = health['species']?.toString();
    final decision = health['decision']?.toString() ?? 'Unknown';
    final decisionConfidence = _asFraction(health['decision_confidence']);
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
            _Badge(
              text: leafType.isEmpty
                  ? 'Leaf'
                  : '${leafType[0].toUpperCase()}${leafType.substring(1)} Leaf',
              color: AppColors.sectionTitle,
            ),
            if (species != null) _Badge(text: species, color: AppColors.titleColor),
            _Badge(text: decision, color: _decisionColor(decision)),
          ],
        ),
        const SizedBox(height: 14),
        _ConfidenceMeter(label: 'Leaf-type routing confidence', fraction: routeConfidence),
        const SizedBox(height: 10),
        _ConfidenceMeter(label: 'Health decision confidence', fraction: decisionConfidence),
        const SizedBox(height: 16),
        Row(
          children: [
            Expanded(child: _StatTile(label: 'Health Value', value: _formatValue(healthValue))),
            const SizedBox(width: 12),
            Expanded(child: _StatTile(label: 'Severity Score', value: _formatValue(severityRaw))),
          ],
        ),
        if (breakdown.isNotEmpty) ...[
          const SizedBox(height: 16),
          const Text(
            'Breakdown',
            style: TextStyle(
              fontWeight: FontWeight.w600,
              color: AppColors.sectionTitle,
              fontSize: 13,
            ),
          ),
          const SizedBox(height: 6),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            decoration: BoxDecoration(
              color: AppColors.outputBg,
              borderRadius: BorderRadius.circular(8),
            ),
            child: Column(
              children: breakdown.entries
                  .map((e) => _KeyValueRow(label: _titleCase(e.key), value: _formatValue(e.value)))
                  .toList(),
            ),
          ),
        ],
      ],
    );
  }
}