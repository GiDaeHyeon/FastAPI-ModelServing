import os
from fastapi import FastAPI, Request, Query
from .model import SentimentCLF


app = FastAPI()
model = SentimentCLF(ckpt_dir=os.environ['CKPT_DIR'])


@app.post('/sentiment')
async def sentiment_clf(request: Request) -> dict:
    req = await request.json()
    text = req['text']
    sentiment = model.sentiment_clf(text)
    return {'sentiment': sentiment}
