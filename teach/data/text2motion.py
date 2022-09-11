from torch.utils.data import Dataset


class Text2Motion(Dataset):
    def __init__(self, texts, lengths):
        if not isinstance(lengths, list):
            raise NotImplementedError("Texts and lengths should be batched.")

        self.texts = texts
        self.lengths = lengths

        self.N = len(self.lengths)


    def __getitem__(self, index):
        return {"text": self.texts[index],
                "length": self.lengths[index]}

    def __len__(self):
        return self.N

    def __repr__(self):
        return f"Text2Motion dataset: {len(self)} data"
