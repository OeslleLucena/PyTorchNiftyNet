import torch
from torch.utils.data import Dataset

class DatasetNiftySampler(Dataset):
    def __init__(self, sampler):
        super(DatasetNiftySampler, self).__init__()
        self.sampler = sampler

    def __getitem__(self, index):
        data = self.sampler(idx=index)
        return torch.from_numpy(data['image'][..., 0, 0]).float(),\
               torch.from_numpy(data['label'][..., 0, 0]).float()

    def __len__(self):
        return len(self.sampler.reader.output_list)



#
# class DatasetNiftyGridSampler(Dataset):
#     def __init__(self, sampler):
#         super(DatasetNiftyGridSampler, self).__init__()
#         self.sampler = sampler
#         self.data_generator = iter(self.sampler())
#
#     def __getitem__(self, index): #ignoring for niftynet
#
#         data = next(self.data_generator)
#         return torch.from_numpy(data['image'][..., 0, 0]).float()
#
#     def __len__(self):
#         return 144
