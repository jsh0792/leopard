import torch
import numpy as np
import torch_geometric
from torch.utils.data.dataloader import default_collate

def collate_MT(batch):
    for item in batch:
        if torch.is_tensor(item[0]):
            img = torch.cat([item[0] for item in batch], dim = 0)
        else:
            img = item[0]

    label_surv = torch.LongTensor([item[1] for item in batch])
    event_time = torch.LongTensor([item[2] for item in batch])
    # event_time = np.array([item[2] for item in batch])
    c = torch.FloatTensor([item[3] for item in batch])
    case_id = np.array([item[4] for item in batch])
    return [img, label_surv, event_time, c, case_id]


# def collate_MT_Graph(batch):
#     elem = batch[0]
#     elem_type = type(elem)        
#     transposed = zip(*batch)

#     label_surv = torch.LongTensor([item[1] for item in batch])
#     event_time = np.array([item[2] for item in batch])
#     c = torch.FloatTensor([item[3] for item in batch])
#     case_id = np.array([item[4] for item in batch])
#     return [samples[0] if isinstance(samples[0], torch_geometric.data.Batch) for samples in transposed, label_surv, event_time, c, case_id]
#     return [samples[0] if isinstance(samples[0], torch_geometric.data.Batch) else default_collate(samples) for samples in transposed, label_surv, event_time, c, case_id]

