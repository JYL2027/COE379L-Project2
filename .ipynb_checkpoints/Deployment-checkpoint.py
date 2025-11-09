from flask import Flask, request
import tensorflow as tf
import numpy as np
import datetime
from flask import jsonify
from io import BytesIO
from PIL import Image
app = Flask(__name__)

model = tf.keras.models.load_model("/model/alternate_lenet5_model.keras")

@app.route('/summary', methods=['GET'])
def model_summary():

    Metadata = {
        "model_name": model.name,
        "framework": "TensorFlow/Keras",
        "framework_version": tf.__version__,
        "created_on": datetime.datetime.now().isoformat(),
        "input_shape": [None] + list(model.input_shape[1:]),
        "num_parameters": model.count_params(),
        "layers": [layer.__class__.__name__ for layer in model.layers],
        "optimizer": model.optimizer._name if hasattr(model, "optimizer") else None,
        "loss_function": model.loss if hasattr(model, "loss") else None,
        "trainable_params": sum(tf.size(v).numpy() for v in model.trainable_weights),
        "non_trainable_params": sum(tf.size(v).numpy() for v in model.non_trainable_weights),
    }

    return jsonify(Metadata)


def preprocessing(image_bytes):

    try:

        # Read binary image from request
        image_bytes = request.data
    
        # Convert to 'L' (grayscale) instead of 'RGB'
        image = Image.open(BytesIO(image_bytes)).convert('L')

        image = image.resize((128, 128))
        image_array = np.array(image) / 255.0
    
        # Reshape to (1, 128, 128, 1)
        image_array = image_array.reshape((1, 128, 128, 1))

        return image_array
        
    except Exception:
        return jsonify({"error": "Invalid image data"})

@app.route('/inference', methods=['POST'])
def inference():
    try:
        if request.files:
            image_bytes = request.files['image'].read()
        else:
            image_bytes = request.data
            
        image_array = preprocessing(image_bytes)

        if image_array is None:
            return jsonify({"error": "Invalid image data"}), 400
        
        # Make prediction
        pred = model.predict(image_array)

        if pred[0][0] > 0.5:
            
            label = "damage"
        else:
            label = "no_damage"

        # Return JSON 
        return jsonify({"prediction": label})

    except Exception as e:
        print(f"An error occurred: {e}") 
        return jsonify({"error": str(e)}), 400

# start the development server
if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0')