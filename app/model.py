import torch
from typing import Optional

from NSMC.datamodule import BERTDataset
from NSMC.trainmodule import Classifier


class SentimentCLF:
    def __init__(self,
                 ckpt_dir=None):
        # Model
        model = Classifier()
        self.model = model.load_from_checkpoint(ckpt_dir)
        self.model.eval()

        # Tokenizer
        self.tokenizer = BERTDataset(use_=False).tokenize

    def sentiment_clf(self, x: Optional[str]) -> Optional[int]:
        input_ids, attention_mask, token_type_ids = self.tokenizer(x)
        y_hat = self.model(input_ids[None], attention_mask[None], token_type_ids[None])
        return int(torch.argmax(y_hat[0]).item())

