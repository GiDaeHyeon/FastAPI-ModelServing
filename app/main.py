import osfrom fastapi import FastAPI, Request, Queryfrom model import SentimentCLFapp = FastAPI()model = SentimentCLF(ckpt_dir=os.environ['checkpoint_dir'])@app.get('/')async def sentiment_clf(request: Request,                        text: Query(None)):    sentiment = model.sentiment_clf(text)    return {'sentiment': sentiment}