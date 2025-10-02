import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'services/classifier.dart';

void main() {
  runApp(const MainApp());
}

class MainApp extends StatelessWidget {
  const MainApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: ClassifyScreen(),
    );
  }
}

class ClassifyScreen extends StatefulWidget {
  const ClassifyScreen({super.key});

  @override
  State<ClassifyScreen> createState() => _ClassifyScreenState();
}

class _ClassifyScreenState extends State<ClassifyScreen> {
  final CapsuleClassifier _classifier = CapsuleClassifier();
  final ImagePicker _picker = ImagePicker();
  File? _imageFile;
  List<ClassificationResult>? _results;
  bool _loading = false;
  String? _expectedFromFilename;
  bool? _isCorrect;

  @override
  void initState() {
    super.initState();
    _classifier.load();
  }

  Future<void> _pickFromGallery() async {
    final picked = await _picker.pickImage(source: ImageSource.gallery, maxWidth: 1024);
    if (picked != null) {
      await _classify(File(picked.path));
    }
  }

  Future<void> _captureWithCamera() async {
    final picked = await _picker.pickImage(source: ImageSource.camera, maxWidth: 1024);
    if (picked != null) {
      await _classify(File(picked.path));
    }
  }

  Future<void> _classify(File file) async {
    setState(() {
      _loading = true;
      _imageFile = file;
      _results = null;
      _expectedFromFilename = _inferExpectedFromFilename(file.path);
      _isCorrect = null;
    });
    final results = await _classifier.classifyFile(file);
    if (!mounted) return;
    setState(() {
      _loading = false;
      _results = results;
      _isCorrect = _computeCorrectness(results, _expectedFromFilename);
    });
  }

  String? _inferExpectedFromFilename(String path) {
    final fileName = path.split(Platform.pathSeparator).last.toLowerCase();
    // Try to match any label keyword contained in filename
    for (final label in _classifier.labels) {
      final key = label.toLowerCase().replaceAll(RegExp(r'[^a-z0-9]+'), ' ').trim();
      if (key.isEmpty) continue;
      final parts = key.split(' ').where((p) => p.isNotEmpty).toList();
      final match = parts.every((p) => fileName.contains(p));
      if (match) return label;
    }
    return null;
  }

  bool? _computeCorrectness(List<ClassificationResult>? results, String? expected) {
    if (results == null || results.isEmpty) return null;
    if (expected == null) return null;
    final top = results.first.label.toLowerCase().trim();
    final exp = expected.toLowerCase().trim();
    return top == exp;
  }

  @override
  void dispose() {
    _classifier.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Vitamin Capsule Scanner')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Expanded(
              child: Container(
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey.shade300),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: _imageFile == null
                    ? const Center(child: Text('Pick or capture an image of a capsule'))
                    : ClipRRect(
                        borderRadius: BorderRadius.circular(12),
                        child: Image.file(_imageFile!, fit: BoxFit.cover),
                      ),
              ),
            ),
            const SizedBox(height: 16),
            if (_loading) const LinearProgressIndicator(),
            if (_results != null) ...[
              const SizedBox(height: 8),
              Card(
                elevation: 0,
                color: Colors.grey.shade100,
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      if (_expectedFromFilename != null)
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Expanded(child: Text('Expected: ${_expectedFromFilename!}')),
                            if (_isCorrect != null)
                              Text(
                                _isCorrect! ? 'Correct' : 'Incorrect',
                                style: TextStyle(color: _isCorrect! ? Colors.green : Colors.red),
                              ),
                          ],
                        ),
                      ..._results!
                          .map((r) => Row(
                                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                                children: [
                                  Expanded(child: Text(r.label)),
                                  Text('${(r.confidence * 100).toStringAsFixed(1)}%'),
                                ],
                              ))
                          .toList(),
                    ],
                  ),
                ),
              ),
            ],
            const SizedBox(height: 8),
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _pickFromGallery,
                    icon: const Icon(Icons.photo),
                    label: const Text('Gallery'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _captureWithCamera,
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('Camera'),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
