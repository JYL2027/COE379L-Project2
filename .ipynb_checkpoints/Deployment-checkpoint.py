from flask import Flask, request
import tensorflow as tf
import numpy as np
app = Flask(__name__)

model = tf.keras.models.load_model()
@app.route('/summary', methods=['GET'])
def model_summary():
    model_info = []
    model.summary(print_fn=lambda x: model_info.append(x))
    summary_str = "\n".join(model_info)

    return jsonify({
        "model_name": model.name,
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "summary": summary_str
    })

@app.route('/inference', methods=['POST'])
def inference():
    try:
        # Read binary image from request
        image_bytes = request.data
        image = Image.open(BytesIO(image_bytes)).convert('RGB')

        # Preprocess: resize to model input (150x150)
        image = image.resize((150, 150))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)  # (1, 150, 150, 3)

        # Make prediction
        pred = model.predict(image_array)
        label = "damage" if pred[0][0] > 0.5 else "no_damage"

        # Return JSON response
        return jsonify({"prediction": label})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# start the development server
if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0')