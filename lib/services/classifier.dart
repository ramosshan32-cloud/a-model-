import 'dart:io';
// import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'dart:math' as math;

class ClassificationResult {
  final String label;
  final double confidence;

  ClassificationResult({required this.label, required this.confidence});
}

class CapsuleClassifier {
  tfl.Interpreter? _interpreter;
  List<String> _labels = const [];
  // String? _loadedModelAssetPath; // Reserved for debugging which model path was used
  static const bool _normalizeToMinusOneToOne = false; // Set true if model trained on [-1,1]
  static const double _confidenceThreshold = 0.0; // Raise (e.g. 0.2) to hide very low scores

  Future<void> load() async {
    try {
      // Try common model filenames in order
      final candidateModels = <String>[
        'assets/model/model.tflite',
        'assets/model/model_unquant.tflite',
      ];
      for (final path in candidateModels) {
        try {
          _interpreter = await tfl.Interpreter.fromAsset(path);
          if (kDebugMode) {
            print('Successfully loaded model from: $path');
          }
          // _loadedModelAssetPath = path;
          break;
        } catch (_) {
          // try next
        }
      }
      if (_interpreter == null) {
        throw StateError('No TFLite model found in assets/model/*.tflite');
      }
      // Log tensor metadata for visibility
      try {
        final inputTensor = _interpreter!.getInputTensor(0);
        final outputTensor = _interpreter!.getOutputTensor(0);
        if (kDebugMode) {
          print('Input tensor -> name: \'${inputTensor.name}\', shape: ${inputTensor.shape}, type: ${inputTensor.type}');
          print('Output tensor -> name: \'${outputTensor.name}\', shape: ${outputTensor.shape}, type: ${outputTensor.type}');
        }
      } catch (_) {
        // Best-effort logging only
      }
      final raw = await rootBundle.loadString('assets/labels/labels.txt');
      _labels = raw
          .split('\n')
          .map((e) => e.trim())
          .where((e) => e.isNotEmpty)
          .map((line) {
            // Handle format like "0 appetason" or just "appetason"
            final parts = line.split(' ');
            return parts.length > 1 ? parts.sublist(1).join(' ') : line;
          })
          .toList();
      if (kDebugMode) {
        print('Loaded labels: $_labels');
      }
    } catch (e) {
      if (kDebugMode) {
        print('Classifier load failed: $e');
      }
      _interpreter = null;
      _labels = const [];
    }
  }

  bool get isLoaded => _interpreter != null && _labels.isNotEmpty;
  List<String> get labels => _labels;

  Future<List<ClassificationResult>> classifyFile(File imageFile) async {
    if (!isLoaded) {
      return _fallbackResults();
    }

    try {
      final bytes = await imageFile.readAsBytes();
      final decoded = img.decodeImage(bytes);
      if (decoded == null) return _fallbackResults();
      // Correct camera EXIF orientation when present
      final oriented = img.bakeOrientation(decoded);

      final inputTensor = _interpreter!.getInputTensor(0);
      final inputShape = inputTensor.shape; // [1, h, w, c]
      final inputH = inputShape.length > 1 ? inputShape[1] : oriented.height;
      final inputW = inputShape.length > 2 ? inputShape[2] : oriented.width;
      final inputC = inputShape.length > 3 ? inputShape[3] : 3;
      // Center-crop to input aspect ratio, then resize
      final targetAspect = inputW / inputH;
      final srcW = oriented.width.toDouble();
      final srcH = oriented.height.toDouble();
      double cropW = srcW;
      double cropH = srcH;
      if ((srcW / srcH) > targetAspect) {
        // Too wide, crop width
        cropW = srcH * targetAspect;
      } else if ((srcW / srcH) < targetAspect) {
        // Too tall, crop height
        cropH = srcW / targetAspect;
      }
      final cropX = ((srcW - cropW) / 2).round();
      final cropY = ((srcH - cropH) / 2).round();
      final cropped = img.copyCrop(
        oriented,
        x: cropX,
        y: cropY,
        width: cropW.round(),
        height: cropH.round(),
      );
      final resized = img.copyResize(cropped, width: inputW, height: inputH);

      // Build input tensor with normalization
      final input = List.generate(
        1,
        (_) => List.generate(
          inputH,
          (y) => List.generate(
            inputW,
            (x) {
              final pixel = resized.getPixel(x, y);
              double r = pixel.r / 255.0;
              double g = pixel.g / 255.0;
              double b = pixel.b / 255.0;
              if (_normalizeToMinusOneToOne) {
                r = (r - 0.5) * 2.0;
                g = (g - 0.5) * 2.0;
                b = (b - 0.5) * 2.0;
              }
              if (inputC == 1) {
                final luminance = 0.299 * r + 0.587 * g + 0.114 * b;
                return [luminance];
              }
              return [r, g, b];
            },
          ),
        ),
      );

      final outputTensor = _interpreter!.getOutputTensor(0);
      final outputShape = outputTensor.shape; // [1, numClasses]
      final numClasses = outputShape.last;
      final output = List.generate(1, (_) => List.filled(numClasses, 0.0));

      _interpreter!.run(input, output);

      final scoresDynamic = output[0] as List;
      List<double> scores = scoresDynamic
          .map((e) => e is int ? e.toDouble() : (e as double))
          .toList();

      // Apply softmax if values look like logits (contain negatives or sum >> 1)
      final sumScores = scores.fold<double>(0.0, (a, b) => a + b);
      final hasNegative = scores.any((v) => v < 0);
      if (hasNegative || sumScores > 1.2) {
        scores = _softmax(scores);
      }

      final int limit = scores.length < _labels.length ? scores.length : _labels.length;
      final results = List.generate(limit, (i) => ClassificationResult(label: _labels[i], confidence: scores[i]))
          .where((r) => r.confidence >= _confidenceThreshold)
          .toList();
      results.sort((a, b) => b.confidence.compareTo(a.confidence));
      return results.take(3).toList();
    } catch (e) {
      if (kDebugMode) {
        print('Classification failed: $e');
      }
      return _fallbackResults();
    }
  }

  List<double> _softmax(List<double> logits) {
    final maxLogit = logits.reduce((a, b) => a > b ? a : b);
    final expVals = logits.map((v) => math.exp(v - maxLogit)).toList();
    final sumExp = expVals.fold<double>(0.0, (a, b) => a + b);
    if (sumExp == 0) {
      return List<double>.filled(logits.length, 0.0);
    }
    return expVals.map((v) => v / sumExp).toList();
  }

  

  List<ClassificationResult> _fallbackResults() {
    return [
      ClassificationResult(label: 'Model not available', confidence: 1.0),
    ];
  }

  void dispose() {
    _interpreter?.close();
  }
}


