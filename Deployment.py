from flask import Flask, request
import tensorflow as tf
import numpy as np
app = Flask(__name__)

model = tf.keras.models.load_model("Project2/COE379L-Project2/alternate_lenet5_model.keras")
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
    
        # 1. Convert to 'L' (grayscale) instead of 'RGB'
        image = Image.open(BytesIO(image_bytes)).convert('L')

        image = image.resize((128, 128))
        image_array = np.array(image) / 255.0
    
        # 2. Reshape to (1, 128, 128, 1)
        # When you convert to 'L', np.array(image) has shape (128, 128)
        # We need to add the batch dimension (1) AND the channel dimension (1)
        image_array = image_array.reshape((1, 128, 128, 1))

        # Make prediction
        pred = model.predict(image_array)
        label = "damage" if pred[0][0] > 0.5 else "no_damage"

        # Return JSON response
        return jsonify({"prediction": label})

    except Exception as e:
        # Optional: Log the actual error to your console for debugging
        print(f"An error occurred: {e}") 
        return jsonify({"error": str(e)}), 400

# start the development server
if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0')