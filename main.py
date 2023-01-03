from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from inference import responseChat
import nltk
import os
nltk.download('punkt')

from load_model import model_load
model = model_load("pkl/GNB_model.sav")

app = FastAPI()
origins = [
    "http://127.0.0.1:8000",
    "http://127.0.0.1:5500",
    "http://id.000webhost.com",
    "https://id.000webhost.com",
    "http://tsunderesimulatorchat.000webhostapp.com",
    "https://tsunderesimulatorchat.000webhostapp.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

class BodyRequest(BaseModel):
    chat_request: str

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    return templates.TemplateResponse("index.html", {"request": request, "message": "Oke"})


@app.post("/responseChat")
async def say_hello(body: BodyRequest):
    res = responseChat(sentence=body.chat_request, model=model)
    return {"message": res}
