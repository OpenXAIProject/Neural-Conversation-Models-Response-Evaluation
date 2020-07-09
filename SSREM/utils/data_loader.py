from torch.utils.data import Dataset, DataLoader


class UttersDataset(Dataset):
    def __init__(self, utters, data=None):
        self.utters = utters

        self.data = data
        self.len = len(utters)   # total_data_size

    def __getitem__(self, index):
        """Return Single data sentence"""
        utter = self.utters[index]

        return utter

    def __len__(self):
        return self.len


def get_loader(utters, batch_size=100, data=None):
    def collate_fn(data):
        contexts, res_true, res_ns1, res_ns2, res_ns3, res_ns4 = zip(*data)

        return contexts, res_true, res_ns1, res_ns2, res_ns3, res_ns4

    dataset = UttersDataset(utters, data=data)

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return data_loader


def get_loader2(utters, batch_size=100, data=None):
    def collate_fn(data):
        contexts, generated_utter, res_true = zip(*data)

        return contexts, generated_utter, res_true

    dataset = UttersDataset(utters, data=data)

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return data_loader
