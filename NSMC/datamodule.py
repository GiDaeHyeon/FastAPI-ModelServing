import torch
from torch.utils.data import Dataset, DataLoader

from pytorch_lightning import LightningDataModule

from transformers import BertTokenizerFast


class BERTDataset(Dataset):
    def __init__(self,
                 data_dir='./dataset',
                 weight='kykim/bert-kor-base',
                 mode='train',
                 max_len=256,
                 use_=True):
        super().__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained(weight)
        if use_:
            with open(f'{data_dir}/ratings_{mode}.txt', 'r') as d:
                self.data = d.readlines()[1:]
                d.close()

        self.max_len = max_len
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def tokenize(self, item):
        encoded_sent = self.tokenizer.encode_plus(text=item,
                                                  add_special_tokens=True,
                                                  max_length=self.max_len,
                                                  truncation=True,
                                                  pad_to_max_length=True,
                                                  return_attention_mask=True,
                                                  return_token_type_ids=True
                                                  )
        input_ids = torch.tensor(encoded_sent.get('input_ids'))
        attention_masks = torch.tensor(encoded_sent.get('attention_mask'))
        token_type_ids = torch.tensor(encoded_sent.get('token_type_ids'))
        return input_ids, attention_masks, token_type_ids

    def __getitem__(self, idx):
        item, label = self.data[idx].split('\t')[1:]
        input_ids, attention_masks, token_type_ids = self.tokenize(item)
        return input_ids, attention_masks, token_type_ids, torch.tensor(int(label)).long()


class BERTDataModule(LightningDataModule):
    def __init__(self,
                 weight='kykim/bert-kor-base',
                 batch_size=32,
                 num_workers=4,
                 max_len=256):
        super().__init__()
        self.trainset = BERTDataset(weight=weight,
                                    max_len=max_len)
        self.valset = BERTDataset(weight=weight,
                                  mode='test',
                                  max_len=max_len)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(dataset=self.trainset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          pin_memory=True,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.valset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          pin_memory=True,
                          drop_last=True)
