from tensorflow.keras.models import model_from_json
import numpy as np
import io
from PIL import Image
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

# Unpacking the model
with open("model.json", "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)
model.load_weights("emotion.h5")

disease_labels = [
    ["Bacterial Blight", "Spreads via rain, irrigation water, wind", "Avoid excessive nitrogen fertilizers"],
    ["Blast", "Spreads through spores in air", "Proper plant spacing"],
    ["Brown Spot", "Drought stress", "Use certified disease-free seeds"],
    ["Tungro", "Virus transmitted by insect vectors", "Use resistant rice varieties"]]


@app.get("/")
def Rice():
    return {"Project": "Anay sir's rice disease prediction"}


@app.get("/health")
def status():
    return {"status": "Up and running"}


def imageTransform(content: bytes):
    # Open image using PIL
    image = Image.open(io.BytesIO(content))

    # Ensure RGB
    image = image.convert("RGB")

    # Resize to 96x96
    image = image.resize((96, 96))

    # Convert to numpy array
    img_array = np.array(image)

    # Normalize
    img_array = img_array.astype("float32") / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


@app.post("/predict")
async def predictRiceDisease(file: UploadFile = File(...)):
    # Read image as bytes
    contents = await file.read()

    # Getting back pre-processed image in form of array
    img_array = imageTransform(contents)

    # Prediction
    pred = model.predict(img_array)[0]
    predicted_index = np.argmax(pred)

    disease_tag = disease_labels[predicted_index][0]
    confidence = int(pred[predicted_index] * 100)

    return {
        "disease": disease_tag,
        "confidence": confidence,
        "cause": disease_labels[predicted_index][1],
        "prevention": disease_labels[predicted_index][2],
    }
