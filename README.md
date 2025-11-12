Project Overview

This project builds and deploys convolutional neural networks (CNNs) to classify post-Hurricane Harvey satellite images into damaged or non-damaged building categories.

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

### Build the Docker Image

Build using the Docker Compose.

`docker-compose build`

### Run the Inference Server

`docker-compose up`


### API Endpoints

- GET /summary: This endpoint returns metadata about the deployed Model

`curl localhost:5000/summary`

- POST /inference: Classifies a provided image as damage or no_damage.

`curl -X POST localhost:5000/inference "file.jpg"`

### Stop The Server

- When done with the server here is how to stop everything

`docker-compose down`
