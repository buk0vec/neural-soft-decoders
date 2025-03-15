import torch
import numpy as np
import galois

GF = galois.GF(2)

def load_pc_matrix(f):
    pc = []
    with open(f, mode="r") as file:
        for l in file:
            pc.append([int(x) for x in l.strip().split(" ")])
    return np.array(pc)

# Gets the generator matrix through the null space of H
def generator_from_nullspace(H):
    H_ = GF(H)
    assert np.linalg.matrix_rank(H_) == min(H.shape), "ERROR: H isn't full rank, cannot get G from Ker(H)"
    return np.array(H_.null_space().T)

def bpsk(codewords):
    return np.array(codewords) * -2 + 1

# binarizes bpsk signal
def binarize(signal):
    return torch.floor((torch.sign(-1 * signal) + 1) / 2)

def awgn(X, ebn0, rate):
    var = ebn0_to_variance(ebn0, rate)
    Z = np.random.normal(scale=np.sqrt(var), size=X.shape)
    Y = X + Z
    return Y

# modern coding theory page 176. factor of 2 is from double sided power spectral density
def ebn0_to_variance(ebn0, rate):
    return 1 / (2 * rate * (10 ** (ebn0/10)))

# Artemasov et. al.
def soft_syndrome(H, gamma):
    s = np.prod(np.repeat(np.sign(gamma)[:,np.newaxis], H.shape[0], axis=1), where=(H.T == 1), axis=0)
    s *= np.min(np.repeat(np.abs(gamma)[:,np.newaxis], H.shape[0], axis=1), where=(H.T == 1), axis=0, initial=np.inf)
    return s

def BER(x_pred, x_gt):
    return torch.mean((x_pred != x_gt).float()).item()

def FER(x_pred, x_gt):
    return torch.mean(torch.any(x_pred != x_gt, dim=-1).float()).item()

class LLRCodeDataset(torch.utils.data.Dataset):
    def __init__(self, file, split="train", EbN0_min=0, EbN0_max=3, epoch_len=1000):
        super().__init__()
        self.H = load_pc_matrix(file)
        self.G = generator_from_nullspace(self.H)
        self.n = self.H.shape[1]
        self.k = self.n - self.H.shape[0]
        self.rate = self.k/self.n
        self.split = split
        self.EbN0_min = EbN0_min
        self.EbN0_max = EbN0_max
        self.epoch_len = epoch_len
    def __len__(self):
        return self.epoch_len
    def __getitem__(self, index):
        if self.split == "train":
            x = np.zeros(self.k)
            c = np.zeros(self.n)
        else:
            x = np.random.randint(0, 2, size=self.k)
            c = (self.G @ x).T % 2
        y = bpsk(c)
        if self.split == "test":
            EbN0 = self.EbN0_min
        else:
            EbN0 = np.random.uniform(self.EbN0_min, self.EbN0_max)
        var = ebn0_to_variance(EbN0, self.rate)
        y_ = awgn(y, EbN0, self.rate)
        llr = y_ * 2 / var
        s = soft_syndrome(self.H, llr)
        return torch.tensor(llr, dtype=torch.float32), torch.tensor(s, dtype=torch.float32), torch.tensor(c, dtype=torch.float32)