import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
import numpy


class MELDRobertaDataset(Dataset):

    def __init__(self, split):
        self.speakers, self.emotion_labels, self.sentiment_labels,\
        self.eroberta1, self.eroberta2, self.eroberta3, self.eroberta4,\
        self.sroberta1, self.sroberta2, self.sroberta3, self.sroberta4,\
        self.videoAudio, self.videoVisual,\
        self.sentences, self.train_ids, self.test_ids, self.valid_ids\
            = pickle.load(open("../data/meld/meld_emotion_semantic_features_roberta.pkl", 'rb'), encoding='latin1')

        if split == 'train':
            self.keys = [x for x in self.train_ids]
        elif split == 'test':
            self.keys = [x for x in self.test_ids]
        elif split == 'valid':
            self.keys = [x for x in self.valid_ids]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(numpy.array(self.eroberta1[vid])),\
               torch.FloatTensor(numpy.array(self.sroberta1[vid])),\
               torch.FloatTensor(numpy.array(self.videoVisual[vid])),\
               torch.FloatTensor(numpy.array(self.videoAudio[vid])),\
               torch.FloatTensor(numpy.array(self.speakers[vid])),\
               torch.FloatTensor(numpy.array([1]*len(self.emotion_labels[vid]))),\
               torch.LongTensor(numpy.array(self.emotion_labels[vid])),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<5 else pad_sequence(dat[i], True) if i<7 else dat[i].tolist() for i in dat]
