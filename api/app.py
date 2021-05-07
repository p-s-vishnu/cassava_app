# Fast APi Imports
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Cassava Imports
from cassava.pretrained import get_model

# T0 process Image
from PIL import Image
import numpy as np


def load_model(name):
    model = get_model(name=name)
    return model


model = load_model("tf_efficientnet_b4")

# Initializing the Fast API
app = FastAPI(
    title="Cassava Classifier",
    version="0.0.2",
    description="To classify cassava leaf dissease type",
)

# Adding CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    image = Image.open(file.file)
    image = np.array(image)
    value = model.predit_as_json(image)
    return {"class_name": value["class_name"], "confidence": str(value["confidence"])}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8003, reload=True)
