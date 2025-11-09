FROM python:3.12

RUN pip install --no-cache-dir tensorflow==2.15 Flask==3.0 Pillow numpy

# Copy model and Flask app
RUN mkdir /app
WORKDIR /app
COPY alternate_lenet5_model.keras /model/alternate_lenet5_model.keras
COPY Deployment.py /app/api.py

# Start Flask server
CMD ["python", "api.py"]
