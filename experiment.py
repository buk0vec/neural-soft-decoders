import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from coding import LLRCodeDataset, binarize, FER, BER
from models import BaselineTransformer, BaselineRNN, BaselineGRU
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import os

torch.manual_seed(42)
np.random.seed(42)
torch.mps.manual_seed(42) # i am gpu poor and my laptop chassis has melted to my desk

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="rnn")
    parser.add_argument("--learning-rate", default=1e-4, type=float)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--epoch-len", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mask", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--code", type=str, required=True)
    parser.add_argument("--transformer-dim", type=int, default=32)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--snr-min", type=float, default=2)
    parser.add_argument("--snr-max", type=float, default=7)
    args = parser.parse_args()

    epochs = args.epochs
    batch_size= args.batch_size
    epoch_len = args.epoch_len
    learning_rate = args.learning_rate
    code = args.code
    
    if not code.endswith(".txt"):
        code += '.txt'

    if not os.path.exists(f"codes/{code}"):
        print(f"ERROR -- please specify a valid code, path 'codes/{code}' does not exist")
        os._exit(1)
        
    dataset = LLRCodeDataset(f"codes/{code}", split="train", epoch_len=batch_size * epoch_len, EbN0_min=args.snr_min, EbN0_max=args.snr_max) 
    test_dataset = LLRCodeDataset(f"codes/{code}", split="val", epoch_len=100, EbN0_min=args.snr_min, EbN0_max=args.snr_max)
    
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

    n, k = dataset.n, dataset.k

    if args.model == "rnn":
        model = BaselineRNN(n=n, k=k, d_expand=5, depth=4, T=5)
    elif args.model == "gru":
        model = BaselineGRU(n=n, k=k, d_expand=5, depth=4, T=5)
    elif args.model == "transformer":
        model = BaselineTransformer(n=n, k=k, n_blocks=args.transformer_layers, d=args.transformer_dim, ecct_mask=None if not args.mask else dataset.H)
    else:
        print(f"ERROR -- {args.model} is not a valid option, please use 'rnn', 'gru', or 'transformer'")
        os._exit(0)
    
    model = model.to(device)

    if args.model == 'transformer':
        model_id = f'experiment_{code[:-4]}_{args.model}_d={args.transformer_dim}_{"mask" if args.mask else "nomask"}_lr={args.learning_rate}_epochs={args.epochs}_noreg'
    else:
        model_id = f'experiment_{code[:-4]}_{args.model}_lr={args.learning_rate}_epochs={args.epochs}'
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, num_workers=0)
    bce_loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-2 * learning_rate)

    model.train()

    writer = SummaryWriter(log_dir=f'runs/experiments/{model_id}')
    print("Starting training for model: ", model_id)
    
    for i in range(epochs):
        running_loss = 0.0
        running_fer = 0.0
        running_ber = 0.0
        if i % 10 == 0:
            print(f"---- Epoch {i} ----")
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

    torch.save(model.state_dict(), f'models/{model_id}.pth')
    print("Done training!")
