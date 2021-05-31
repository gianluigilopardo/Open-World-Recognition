import numpy as np
import torch
from PIL import Image

import utils
import params


class Subset(Dataset):
    """
    serve per prendere dei sottoinsiemi dell'insieme originale a seconda del task in cui ci troviamo
    quindi anzichè fare il training su tutto il dataset lo faccio sulle immagini le cui classi sono
    nel task in cui ci troviamo

    Noi dobbiamo riaddestrare tutta la nostra rete, non su tutto il dataset ma su dei task dei dataset,
    dei batch di dataset. Ad esempio se andiamo di 10 in 10, il primo task ha dimensione 10e il primo
    subset sarà un sottoinsieme di 10 classi del dataset

    """
    def __init__(self, dataset, indices, transform):
        """
        :param dataset: the whole dataset
        :param indices: indices to take and put in the subset
        :param transform: transformers che gli passiamo, serve quando creaimo il dataset se voglio usare transformer
        """
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        """
        dato un insieme di indici lui mi restituisce le immagini e le labels delle immagini
        :param idx:
        :return:
        """
        image, labels, _ = self.dataset[self.indices[idx]]
        return self.transform(Image.fromarray(np.transpose(image))), labels, idx

    def __len__(self):
        return len(self.indices)
