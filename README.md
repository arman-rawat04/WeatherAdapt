# WeatherAdapt-STR: Adaptive Scene Text Recognition with Real-time Weather Classification

A comprehensive system that classifies weather conditions from input images and dynamically selects the most appropriate text recognition model or preprocessing pipeline for that specific weather condition.

## Project Structure

```
nlp/
├── config.py                 # Configuration file
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── Dockerfile               # Docker configuration
├── notebooks/               # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_weather_augmentation.ipynb
│   ├── 03_weather_classification.ipynb
│   ├── 04_text_recognition_training.ipynb
│   ├── 05_adaptive_routing.ipynb
│   ├── 06_inference_evaluation.ipynb
│   └── 07_api_demo.ipynb
├── src/                     # Source code modules
│   ├── __init__.py
│   ├── data_loader.py       # Dataset loaders
│   ├── weather_augmentation.py  # Weather effect simulation
│   ├── models/              # Model architectures
│   │   ├── __init__.py
│   │   ├── weather_classifier.py
│   │   └── text_recognizer.py
│   ├── preprocessing.py     # Image preprocessing
│   └── utils.py            # Utility functions
├── outputs/                 # Generated outputs
│   ├── models/             # Trained models
│   ├── logs/               # Training logs
│   ├── results/            # Evaluation results
│   └── processed_data/    # Processed datasets
└── api/                    # API server
    ├── __init__.py
    ├── main.py            # FastAPI application
    └── inference.py       # Inference logic
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. For GPU support (optional):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Training

1. **Data Exploration**: Run `notebooks/01_data_exploration.ipynb` to explore datasets
2. **Weather Augmentation**: Run `notebooks/02_weather_augmentation.ipynb` to create synthetic weather data
3. **Weather Classification**: Run `notebooks/03_weather_classification.ipynb` to train weather classifier
4. **Text Recognition**: Run `notebooks/04_text_recognition_training.ipynb` to train text recognition models
5. **Adaptive Routing**: Run `notebooks/05_adaptive_routing.ipynb` to train the routing system

### Inference

Run `notebooks/06_inference_evaluation.ipynb` for inference and evaluation.

### API Deployment

1. Start the API server:
```bash
cd api
python main.py
```

2. Or using Docker:
```bash
docker build -t weatheradapt-str .
docker run -p 8000:8000 weatheradapt-str
```

3. Test the API:
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: multipart/form-data" -F "file=@test_image.jpg"
```

## Datasets

- **ICDAR 2015**: Scene text images with annotations
- **RoadText-1K**: Road scene videos with text annotations
- **SynthText**: Synthetic text images
- **Weather6K**: 6862 weather classification images
- **Weather1K**: 1125 weather classification images

## Models

- **Weather Classifier**: EfficientNet-B0 for weather condition classification
- **Text Recognizer**: CRNN or TrOCR for scene text recognition
- **Adaptive Router**: Routes inputs based on weather classification

## License

See individual dataset licenses in their respective directories.

