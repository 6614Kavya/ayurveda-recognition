import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import '../models/picked_image.dart';
import '../utils/format_utils.dart';

class ApiException implements Exception {
  final String message;
  final int? statusCode;
  ApiException(this.message, [this.statusCode]);
  @override
  String toString() => message;
}

class ApiService {
  static const String baseUrl = 'https://ayurveda-recognition.onrender.com';

  static Future<Map<String, dynamic>> checkHealth() async {
    try {
      final res = await http.get(Uri.parse('$baseUrl/health'));
      return jsonDecode(res.body) as Map<String, dynamic>;
    } catch (_) {
      throw ApiException('Could not reach API — is the backend running?');
    }
  }

  static Future<Map<String, dynamic>> predict({
    required String endpoint,
    required PickedImage image,
  }) async {
    final request = http.MultipartRequest(
      'POST',
      Uri.parse('$baseUrl/predict/$endpoint'),
    );
    request.files.add(
      http.MultipartFile.fromBytes(
        'file',
        image.bytes,
        filename: image.filename,
        contentType: MediaType.parse(mimeTypeFor(image.filename)),
      ),
    );
    return _sendAndParse(request);
  }

  static Future<Map<String, dynamic>> predictLeafHealth({
    required PickedImage top,
    required PickedImage bottom,
  }) async {
    final request = http.MultipartRequest(
      'POST',
      Uri.parse('$baseUrl/predict/leaf-health'),
    );
    request.files.add(
      http.MultipartFile.fromBytes(
        'top_file',
        top.bytes,
        filename: top.filename,
        contentType: MediaType.parse(mimeTypeFor(top.filename)),
      ),
    );
    request.files.add(
      http.MultipartFile.fromBytes(
        'bottom_file',
        bottom.bytes,
        filename: bottom.filename,
        contentType: MediaType.parse(mimeTypeFor(bottom.filename)),
      ),
    );
    return _sendAndParse(request);
  }

  static Future<Map<String, dynamic>> _sendAndParse(
    http.MultipartRequest request,
  ) async {
    final streamed = await request.send();
    final res = await http.Response.fromStream(streamed);

    Map<String, dynamic> body;
    try {
      body = jsonDecode(res.body) as Map<String, dynamic>;
    } catch (_) {
      // Server returned non-JSON (e.g. a proxy-level 500/502 HTML/text page)
      final preview =
          res.body.length > 150 ? res.body.substring(0, 150) : res.body;
      throw ApiException('Server error (${res.statusCode}): $preview', res.statusCode);
    }

    if (res.statusCode != 200) {
      throw ApiException(
        body['detail']?.toString() ?? 'Upload failed (${res.statusCode})',
        res.statusCode,
      );
    }
    return body;
  }
}