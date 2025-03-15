import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from coding import LLRCodeDataset, binarize, FER, BER
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR


torch.manual_seed(42)
np.random.seed(42)
torch.mps.manual_seed(42)

writer = SummaryWriter(log_dir='runs/baseline-2db-7db')

# Baseline RNN network that tries to estimate LLR noise through time
# inspired by Artemasov et. al.
class BaselineRNN(nn.Module):
    def __init__(self, n=128, k=64, d_expand=5, depth=4, T=5):
        super().__init__()
        self.input_size = n + (n - k)
        self.rnn = nn.RNN(self.input_size, n * d_expand, depth, batch_first=False)
        self.proj = nn.Linear(n * d_expand * T, n)
        self.T = T

    def forward(self, x, s):
        x_ = torch.concat([torch.abs(x), s], axis=1)
        B = x.shape[0]
        xs = x_.repeat(self.T, 1, 1)
        outputs = self.rnn(xs)[0]
        outputs = outputs.permute(1, 0, 2).flatten(start_dim=1)
        z = self.proj(outputs)
        out = x - torch.sign(x) * z
        return out

if __name__ == "__main__":
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    epochs = 200
    batch_size=128
    epoch_len = 1000
    learning_rate = 1e-4 # prev 1e-4
    
    model = BaselineRNN()
    model = model.to(device)
    
    dataset = LLRCodeDataset("codes/POLAR_N128_K64.txt", split="train", epoch_len=batch_size * epoch_len, EbN0_min=0, EbN0_max=3) # prev default
    test_dataset = LLRCodeDataset("codes/POLAR_N128_K64.txt", split="val", epoch_len=100, EbN0_min=0, EbN0_max=3) # prev default
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, num_workers=0)
    bce_loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-6) # default 5e-6

    model.train()
    
    for i in range(epochs):
        running_loss = 0.0
        running_fer = 0.0
        running_ber = 0.0
        for llr, s, x in tqdm(train_loader):
            optimizer.zero_grad()
            llr, s, x = llr.to(device), s.to(device), x.to(device)
            llr_ = model(llr, s)
            # bce loss on negative llr so positive = 1
            loss = bce_loss(-1 * llr_, x)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            x_ = binarize(llr_)
            running_fer += FER(x_, x)
            running_ber += BER(x_, x)
        for llr, s, x, in test_loader:
            with torch.no_grad():
                llr, s, x = llr.to(device), s.to(device), x.to(device)
                llr_ = model(llr, s)
                x_ = binarize(llr_)
                fer = FER(x_, x)
                ber = BER(x_, x)
                l = bce_loss(-1 * llr_, x).item()
                print("Nonzero loss:", l)
                print("Nonzero FER:", fer)
                print("Nonzero BER:", ber)
                writer.add_scalar('Loss/val', l, i)
                writer.add_scalar('FER/val', fer, i)
                writer.add_scalar('BER/val', ber, i)
        scheduler.step()
        writer.add_scalar('Loss/train', running_loss/epoch_len, i)
        writer.add_scalar('FER/train', running_fer/epoch_len, i)
        writer.add_scalar('BER/train', running_ber/epoch_len, i)
        print("Loss: ", running_loss/epoch_len)
        print("Running FER: ", running_fer/epoch_len)
        print("Running BER: ", running_ber/epoch_len)
    #     run.log({"acc": running_loss, "loss": running_loss/epoch_len})
    # run.finish()
    torch.save(model.state_dict(), 'models/baseline_rnn_2db-7db.pth')
