import os
from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np


from keras.models import load_model
import tensorflow as tf
import base64
from io import BytesIO

import logging
import flask_monitoringdashboard as dashboard

app = Flask(__name__)

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


@app.route('/')
def index():

    return render_template("index.html")


@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    try:
        # chargement modèle deep learning
        model = tf.keras.models.load_model(
            "models/unet_vgg16_categorical_crossentropy_raw_data.keras", compile=False)

        colors = np.array([[68,   1,  84],
                           [70,  49, 126],
                           [54,  91, 140],
                           [39, 126, 142],
                           [31, 161, 135],
                           [73, 193, 109],
                           [159, 217,  56],
                           [253, 231,  36]])

        if request.method == 'POST':
            # récupération de l'image dans le formulaire
            image = request.files['file']

            if image.filename == '':
                return "Nom de fichier invalide"

            # ouverture de l'image grace à pillow avec conversion RGB
            img = Image.open(image)
            if img.mode == 'RGBA':
                img = img.convert('RGB')

            # redimensionnement en 256 * 256 (entrée du modèle)
            IMAGE_SIZE = 256
            img_resized = img.resize(
                (IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.NEAREST)
            img_resized = np.array(img_resized)

            # normalisation (valeurs décimales entre 0 et 1)
            img_resized = np.expand_dims(img_resized, 0)
            img_resized = img_resized / 255.

            # prédiction (génération d'un masque de prédiction)
            predict_mask = model.predict(img_resized, verbose=0)
            predict_mask = np.argmax(predict_mask, axis=3)
            predict_mask = np.squeeze(predict_mask, axis=0)
            predict_mask = predict_mask.astype(np.uint8)
            # création d'une image à partir d'un tableau
            # création image depuis numpy array
            predict_mask = Image.fromarray(predict_mask)
            predict_mask = predict_mask.resize(
                (img.size[0], img.size[1]), resample=Image.Resampling.NEAREST)

            # transformation du masque en image couleur
            predict_mask = np.array(predict_mask)
            predict_mask = colors[predict_mask]
            predict_mask = predict_mask.astype(np.uint8)

            # conversion en base64 de l'image originale
            buffered_img = BytesIO()  # Création objet bytesIO binaire
            # stockage des données binaire de img dans buffered_img
            img.save(buffered_img, format="PNG")
            base64_img = base64.b64encode(
                buffered_img.getvalue()).decode("utf-8")  # binaire à texte (base64)

            # conversion en base 64 du masque
            buffered_mask = BytesIO()  # création object BytesIO binaire
            # stockage des données binaires de predict_mask dans buffered_mask
            predict_mask.save(buffered_mask, format="PNG")
            base64_mask = base64.b64encode(
                buffered_mask.getvalue()).decode("utf-8")  # binaire à texte (base64)

            logging.info("Prediction successful for image: %s", image.filename)

            return jsonify({'message': "predict ok", "img_data": base64_img, "mask_data": base64_mask})

    except Exception as e:
        logging.error("Error occurred: %s", e)
        return jsonify({'message': "Error occurred during prediction", "error": str(e)})


dashboard.config.enable_logging = True
dashboard.bind(app)


if __name__ == '__main__':
    app.run(debug=True, port=8000)
