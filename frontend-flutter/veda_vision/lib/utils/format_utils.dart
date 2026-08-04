double asFraction(dynamic v) {
  final d = (v is num) ? v.toDouble() : double.tryParse(v?.toString() ?? '') ?? 0.0;
  return d > 1.0 ? d / 100.0 : d;
}

double parsePercent(dynamic v) {
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

String titleCase(String s) => s
    .split('_')
    .map((w) => w.isEmpty ? w : '${w[0].toUpperCase()}${w.substring(1)}')
    .join(' ');

String formatValue(dynamic v) {
  if (v is int) return v.toString();
  if (v is double) return v.toStringAsFixed(2);
  return v?.toString() ?? '—';
}

String mimeTypeFor(String filename) {
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