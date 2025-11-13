## Project Overview

This project builds and deploys a convolutional neural network (CNN) to classify post-Hurricane Harvey satellite images into damaged or non-damaged building categories. The goal is to enhance the understanding of hurricane data and to deploy it to unseen satellite images.

## Contents
The project includes:
1. Data preprocessing and visualization
2. Model design, training, and evaluation
3. Model inference server and deployment using Docker
4. A written report summarizing the methodology and results

## Github Repo Structure
```text
├── docker-compose.yml
├── damage
│   ├── picture files of damaged structures
├── no_damage
│   ├── picture files of undamaged structures
├── Dockerfile
├── Deployment.py
├── alternate_lenet5_model.keras
├── Project02_ModelTraining.ipynb
├── Project02_Report.pdf
├── Use_of_AI.md
└── README.md
```

# Setup Instructions
### Clone the Repository
`git clone https://github.com/JYL2027/COE379L-Project2.git`

`cd <COE379L-Project2>`

### Pull Docker Image
Pull the image using `docker pull jyl2027/hurricane-api:latest`

### Build the Docker Image

Build using the Docker Compose.

`docker compose build`

### Run the Inference Server

`docker compose up -d --build`

### API Endpoints

- GET /summary: This endpoint returns metadata about the deployed Model

`curl localhost:5000/summary`

- POST /inference: Classifies a provided image as damaged or not damaged.

`curl -X POST -F "image=@damage/example.jpeg" localhost:5000/inference`

For the `POST` request, replace "damage/example.jpeg" with the location of the image you want to apply inference.

### Stop The Server

- When completed with the inference analysis, please run the following command to remove and stop the container.

`docker compose down`

### Example Executions
Below are example commands and their results using the outline of the previous section.
### Inference
`curl -X POST -F "image=@/home/ubuntu/nb-data/Project2/COE379L-Project2/damage/-93.528502_30.987438.jpeg" http://localhost:5000/inference`

Result: `{
  "prediction": "damage"
}`

### Summary
`curl localhost:5000/summary`

Result: `{
  "input_shape": [
    null,
    128,
    128,
    1
  ],
  "layers": [
    "Conv2D",
    "MaxPooling2D",
    "Conv2D",
    "MaxPooling2D",
    "Conv2D",
    "MaxPooling2D",
    "Conv2D",
    "MaxPooling2D",
    "Flatten",
    "Dropout",
    "Dense",
    "Dense"
  ],
  "loss_function": "binary_crossentropy",
  "model_name": "Damage_Classifier",
  "model_summary": "CNN LeNet-5 Alternative model for classifying Hurrican Harvey building image data (Damage or No Damage)",
  "num_layers": 12,
  "num_parameters": 2600577,
  "optimizer": "adam",
  "output_shape": [
    null,
    1
  ],
  "version": "v1"
}`


