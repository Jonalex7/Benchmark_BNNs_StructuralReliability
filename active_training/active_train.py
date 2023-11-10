import torch

class ActiveTrain():
    def __init__(self):
        print('engine humming')

    def get_active_points(self, x, x_mc, y_mc n_points):
        y_mean, y_std = torch.mean(y_mc, 1), torch.std(y_mc, 1)
        sorted = torch.topk(y_std, int(n_points))
        idx_max_ystd = sorted[1] # Taking the indices of the max std
        x_new = x_mc[idx_max_ystd]
        x_next = torch.cat( (x, x_new))
        return x_next

