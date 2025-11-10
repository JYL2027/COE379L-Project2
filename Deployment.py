from flask import Flask, request
import tensorflow as tf
import numpy as np
from flask import jsonify
from io import BytesIO
from PIL import Image
from io import StringIO
app = Flask(__name__)

model = tf.keras.models.load_model("/model/alternate_lenet5_model.keras")

@app.route('/summary', methods=['GET'])
def model_summary():
    # [2] Chat GPT Use
    stream = StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    summary_str = stream.getvalue().strip()
    
    Metadata = {
        "model_name": model.name,
        "model_summary": summary_str,
        "num_parameters": model.count_params(),
        "num_layers": len(model.layers),
        "layers": [layer.__class__.__name__ for layer in model.layers],
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "optimizer": getattr(model.optimizer, "_name", str(model.optimizer)),
        "loss_function": str(model.loss),
        "activation_functions": list({layer.activation.__name__ for layer in model.layers if hasattr(layer, "activation")})
    }

    return jsonify(Metadata)


def preprocessing(image_bytes):

    try:

        # Read binary image from request
        image_bytes = request.data
    
        # Convert to 'L' (grayscale)
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