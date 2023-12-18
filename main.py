import base64
from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel
import cv2 as cv
from PIL import Image, ImageEnhance, ImageOps
import os, time, sys, shutil
from fastapi import FastAPI, File, Request, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import io
import json
from utills import opResize
from starlette.requests import Request
import os
from counter import scan      # detect/detect_face.py
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def greeting():
      return {'greeting': 'Hello World'}
  
class opResizer(BaseModel):
    filename: str
    img_dimensions: str
    encoded_img: str


@app.post("/scanner")
async def scanner(request: Request, file: bytes = File(...)):
    data = {"success": False}

    if request.method == "POST":
        data = scan(file)

    #return data
    _, encoded_img = cv.imencode('.PNG', data)

    encoded_img = base64.b64encode(encoded_img)

    return{
        'filename': 'b.png',
        'encoded_img': encoded_img,
    }

@app.post("/resize")
async def resize(request: Request, file: bytes = File(...),wd: int = None,hg: int = None,ld: float = None):
    data = {"success": False}

    if request.method == "POST":
        data = opResize(file,wd,hg,ld)

    #return data
    _, encoded_img = cv.imencode('.jpg', data)

    encoded_img = base64.b64encode(encoded_img)

    return{
        'filename': 'r.jpg',
        'encoded_img': encoded_img,
    }

@app.post("/opresize", response_model=opResizer)
async def opresize_route(file: bytes = File(...),wd: int = None,hg: int = None,ld: float = None):
    
    stream = io.BytesIO(file)

    img = np.asarray(bytearray(stream.read()), dtype="uint8")
    img = cv.imdecode(img, cv.IMREAD_COLOR)
    
    #contents = await file.read()
    #nparr = np.frombuffer(contents, np.uint8)
    #img = cv.imdecode(nparr, cv.IMREAD_COLOR)

    img_dimensions = str(img.shape)
    return_img = opResize(img,wd,hg,ld)
    print("bura")
    # line that fixed it
    _, encoded_img = cv.imencode('.PNG', return_img)

    encoded_img = base64.b64encode(encoded_img)

    return{
        'filename': file.filename,
        'dimensions': img_dimensions,
        'encoded_img': encoded_img,
    }
    

# python main.py
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    