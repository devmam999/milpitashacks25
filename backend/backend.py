#The necessary imports, FastAPI for backend functionality,
# UploadFile for handling file uploads, and PIL for image processing.
# numpy is used for numerical operations, and io for handling byte streams. Models is imported to access the emotion analysis functionality.
#FastAPI is for recieving and uploading the images as requests, connecting it with the rest of the world. 
#open_ai_feedback is imported to send feedback to OpenAI for further analysis and improvement of the emotion detection model.
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
import models

# This is the main FastAPI application that serves as the backend for emotion analysis.
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
) 

#Test function to check if the backend is running.
@app.get("/")
def root():
    return {"message": "Backend is running!"}


# Endpoint to analyze emotion from an uploaded image file.
@app.post("/analyze-emotion/")
async def analyze_emotion(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image.convert("RGB"))

        emotion, score, distress = models.analyze_emotion(image_np)

        print("DEBUG distress scale type:", type(distress.get("scale")), distress.get("scale"))
        if isinstance(distress.get("scale"), (list, tuple)):
            print("DEBUG distress scale element type:", type(distress["scale"][0]))

        # Defensive: ensure all distress dict fields are native Python types
        distress_clean = {
            "color": str(distress.get("color")),
            "scale": tuple(int(x) for x in distress.get("scale", (0, 0))),
            "description": str(distress.get("description"))
        }


        return {
            "emotion": str(emotion),
            "score": float(score),
            "distress": distress_clean,
        }
    except Exception as e:
        return {"error": str(e)}
