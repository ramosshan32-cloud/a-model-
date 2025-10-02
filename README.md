# flutter_application_1

A new Flutter project.

## Model & Labels

Place your TensorFlow Lite model and labels here:

- `assets/model/model.tflite`
- `assets/labels/labels.txt`

Create folders and then run:

```bash
mkdir -p assets/model assets/labels
flutter pub get
```

The app will attempt to load these at runtime.