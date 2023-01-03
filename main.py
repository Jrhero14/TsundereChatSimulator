from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from inference import responseChat
import nltk
nltk.download('punkt')

from load_model import model_load
model = model_load("pkl/GNB_model.sav")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

class BodyRequest(BaseModel):
    chat_request: str

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "message": "Oke"})


@app.post("/responseChat")
async def say_hello(body: BodyRequest):
    res = responseChat(sentence=body.chat_request, model=model)
    return {"message": res}
