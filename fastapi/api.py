import os
from fastapi import FastAPI , UploadFile, File
import uvicorn
import numpy as np 
import pandas as pd
import tensorflow as tf 
from io import BytesIO
from PIL import Image

app = FastAPI()
file_path = "C:/Users/Arnav Singh/OneDrive/Desktop/tea leaf/Model/saved_model.keras"
model = tf.keras.models.load_model(file_path)
class_name = ["Anthracnose", "algal leaf","bird eye spot","brown blight","gray light","healthy","red leaf spot","white spot"]




@app.get("/live" )
async def newfunction():
    return {f"running"}





def coverting_to_image_file(bytes) -> np.ndarray:
    image_file = np.array(Image.open( BytesIO( bytes)))
    return image_file


@app.post( "/Predict")
async def Predict(file: UploadFile = File(...)):
    image_array = coverting_to_image_file (await file.read())
    
    Predicted_image =model.predict(np.expand_dims(image_array, 0)) 
    Predicted_class = class_name[np.argmax(Predicted_image[0])]
    confidence = np.max( Predicted_image[0])
    
    return {
        'class' : Predicted_class,
        'confidence ' : float(confidence)
    }
    


if __name__ == "__main__":
    uvicorn.run( app , host="localhost" , port = 8000)