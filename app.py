import os
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config["ALLOWED_EXTENSIONS"] = set(['png', 'jpg', 'jpeg'])
app.config["UPLOAD_FOLDER"] = "uploads/"

def allowed_file(filename):
    return "." in filename and filename.split(".", 1)[1] in app.config["ALLOWED_EXTENSIONS"]

model = load_model("best_model.h5", compile=False)
with open('nutrition_101.csv', 'r') as file:
    reader = pd.read_csv(file)

food_list = [
    'apple pie', 'baby back ribs', 'baklava', 'beef carpaccio', 'beef tartare', 'beet salad', 'beignets', 'bibimbap',
    'bread pudding', 'breakfast burrito', 'bruschetta', 'caesar salad', 'cannoli', 'caprese salad', 'carrot cake',
    'ceviche', 'cheese plate', 'cheesecake', 'chicken curry', 'chicken quesadilla', 'chicken wings', 'chocolate cake',
    'chocolate mousse', 'churros', 'clam chowder', 'club sandwich', 'crab cakes', 'creme brulee', 'croque madame',
    'cup cakes', 'deviled eggs', 'donuts', 'dumplings', 'edamame', 'eggs benedict', 'escargots', 'falafel',
    'filet mignon', 'fish and_chips', 'foie gras', 'french fries', 'french onion soup', 'french toast',
    'fried calamari', 'fried rice', 'frozen yogurt', 'garlic bread', 'gnocchi', 'greek salad',
    'grilled cheese sandwich', 'grilled salmon', 'guacamole', 'gyoza', 'hamburger', 'hot and sour soup', 'hot dog',
    'huevos rancheros', 'hummus', 'ice cream', 'lasagna', 'lobster bisque', 'lobster roll sandwich',
    'macaroni and cheese', 'macarons', 'miso soup', 'mussels', 'nachos', 'omelette', 'onion rings', 'oysters',
    'pad thai', 'paella', 'pancakes', 'panna cotta', 'peking duck', 'pho', 'pizza', 'pork chop', 'poutine',
    'prime rib', 'pulled pork sandwich', 'ramen', 'ravioli', 'red velvet cake', 'risotto', 'samosa', 'sashimi',
    'scallops', 'seaweed salad', 'shrimp and grits', 'spaghetti bolognese', 'spaghetti carbonara', 'spring rolls',
    'steak', 'strawberry shortcake', 'sushi', 'tacos', 'octopus balls', 'tiramisu', 'tuna tartare', 'waffles'
]

def convert_class_name(name):
    converted_name = name.replace("_", " ")
    return converted_name

def predict_class_with_nutrition(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.

    prediction = model.predict(img)
    index = np.argmax(prediction)

    if 0 <= index < len(food_list):
        converted_class_name = convert_class_name(food_list[index])

        nutrition_values = reader[reader['name'] == converted_class_name]
        accuracy_score = prediction[0][index]

        return {
            "food_name": converted_class_name,
            "accuracy": float(accuracy_score),
            "nutrition_values": {
                "protein": nutrition_values['protein'].values[0],
                "calcium": nutrition_values['calcium'].values[0],
                "fat": nutrition_values['fat'].values[0],
                "carbohydrates": nutrition_values['carbohydrates'].values[0],
                "vitamins": nutrition_values['vitamins'].values[0]
            }
        }
    else:
        return {
            "error": f"Warning: Index {index} out of range for food_list."
        }

@app.route("/")
def index():
    return jsonify({
        "status": {
            "code": 200,
            "message": "Success fetching the API",
        },
        "data": None
    }), 200

@app.route("/prediction", methods=["POST"])
def prediction():
    if request.method == "POST":
        image = request.files["image"]
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            result = predict_class_with_nutrition(image_path)

            return jsonify({
                "status": {
                    "code": 200,
                    "message": "Success predicting",
                },
                "data": result
            }), 200
        else:
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "Client side error"
                },
                "data": None
            }), 400
    else:
        return jsonify({
            "status": {
                "code": 405,
                "message": "Method not allowed"
            },
            "data": None,
        }), 405

if __name__ == "__main__":
    app.run(debug=True)
