# Object Detection System
Production-oriented computer vision application for object detection using deep learning, TensorFlow, Flask, and Docker. The project demonstrates the development of an end-to-end object detection workflow covering data processing, model training, inference serving, and deployment preparation.

Designed to explore practical AI deployment workflows and reproducible machine learning engineering practices, the system combines model development with API integration and containerised deployment.

# Overview
This repository contains an object detection pipeline built to support image-based object recognition and inference workflows using deep learning models.
The project focuses on:
- scalable model training workflows
- inference serving
- reproducible environments
- deployment readiness
- practical computer vision engineering

The implementation follows a structured machine learning workflow from dataset preparation through deployment-oriented integration.

# Key Features
- Deep learning-based object detection
- TensorFlow training and inference pipeline
- Flask API for serving predictions
- Docker containerisation for reproducibility
- Modular project structure
- Experimentation and validation workflow
- Deployment-ready environment setup

# Technology Stack
| Area | Technologies |
|---|---|
| Programming | Python |
| Machine Learning | TensorFlow, Keras |
| Computer Vision | OpenCV |
| Backend/API | Flask |
| Deployment | Docker |
| Data Processing | NumPy, Pandas |
| Development Tools | Jupyter Notebook, Git |

## Repository Structure
```bash
Object_detection/
│
├── app/                 # Flask application and inference API
├── data/                # Dataset and preprocessing resources
├── models/              # Trained models and saved checkpoints
├── notebooks/           # Experimentation and analysis notebooks
├── src/                 # Core training, inference, and utility scripts
├── Dockerfile           # Docker container configuration
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
```

# Installation
Clone the repository:
```bash
git clone https://github.com/ryzzo/Object_detection.git cd Object_detection 
```

Create and activate a virtual environment:
```bash 
python -m venv venv
```

### Windows
```bash
venv\Scripts\activate 
```

### Linux / macOS
```bash
source venv/bin/activate 
```

Install dependencies:
```bash
pip install -r requirements.txt 
```

# Running the Application
## Train Model
```bash
python train.py 
```

## Run Inference
```bash
python inference.py 
```

## Start Flask API
```bash
python app.py 
```

# Docker Deployment
Build Docker image:
```bash
docker build -t object-detection-system . 
```

Run container:
```bash
docker run object-detection-system 
```

# Workflow
The project follows a structured machine learning engineering pipeline:
1. Dataset preparation and preprocessing  
2. Model training and validation  
3. Inference testing  
4. API integration using Flask  
5. Containerisation using Docker  
6. Deployment-oriented environment configuration  

# Engineering Focus
This project was developed to strengthen practical understanding of:
- production-oriented machine learning workflows
- deployment and reproducibility
- computer vision system integration
- model serving and inference pipelines
- applied AI engineering practices

The implementation prioritises maintainability, modularity, and reproducible experimentation workflows.

# Future Enhancements
Potential future improvements include:
- real-time streaming inference
- experiment tracking integration
- cloud deployment
- model performance optimisation
- monitoring and logging
- CI/CD integration
- frontend dashboard support

# Author

Brian Rabary Orimba  
GitHub: https://github.com/ryzzo
