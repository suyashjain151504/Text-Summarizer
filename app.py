from fastapi import FastAPI
import uvicorn
import sys
import os
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from textSummarizer.pipeline.prediction import PredictionPipeline
import subprocess



text: str = "What is text summarization?"

app = FastAPI()

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")




@app.get("/train")
async def training():
    try:
        subprocess.run("python main.py", shell=True)
        return Response("Training Successful !!")
                        
    except Exception as e:
        return Response(f"Error occurred: {e}")
    

@app.get("/predict")
async def predict_route(text):
    try:
        obj = PredictionPipeline()
        text = obj.predict(text)
        return text
    except Exception as e:
        raise e
    
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)