import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from coding import LLRCodeDataset, binarize, FER, BER
from models import BaselineTransformer, BaselineRNN, BaselineGRU
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    torch.mps.manual_seed(42)
    
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", '-m', action='extend', type=str, nargs='+', required=True)
    parser.add_argument("--eval-points", '-e', action='extend', type=float, nargs='+', default=[4, 5, 6])
    parser.add_argument("--snr-min", type=float, default=0)
    parser.add_argument("--snr-max", type=float, default=8)
    parser.add_argument("--n-spaces", type=int, default=9)
    parser.add_argument("--n-samples", type=int, default=2000)
    args = parser.parse_args()
    
    code = '_'.join(args.models[0].split("_")[1:4])
    
    if not code.endswith(".txt"):
        code_path = code + '.txt'
    else:
        code_path = code
    
    models = []
    model_names = []
    dataset = LLRCodeDataset(f"codes/{code_path}", split="test", epoch_len=args.n_samples, EbN0_min=0, EbN0_max=0)
    n, k = dataset.n, dataset.k
    for model in args.models:
        if not model.endswith(".pth"):
            model += '.pth'
        if not os.path.exists(f"models/{model}"):
            print(f"ERROR -- please specify a valid model, path 'models/{model}' does not exist")
            os._exit(1)
        if code not in model:
            print(f"ERROR -- please make sure all models use same code, path 'models/{model}' does not contain code '{code}'")
            os._exit(1)
        if 'rnn' in model:
            m = BaselineRNN(n=n, k=k, d_expand=5, depth=4, T=5)
            model_names.append("RNN")
        elif 'gru' in model:
            m = BaselineGRU(n=n, k=k, d_expand=5, depth=4, T=5)
            model_names.append("GRU")
        elif 'transformer' in model:
            m = BaselineTransformer(n=n, k=k, n_blocks=2, d=32, ecct_mask=None if not 'mask' in model else dataset.H)
            model_names.append("Transformer" if 'nomask' in model else "Transformer (masked)")
        m.load_state_dict(torch.load(f"models/{model}", map_location=device), strict=False)
        m.eval()
        m = m.to(device)
        models.append(m)
    
    # SNR curve generation
    snrs = np.linspace(args.snr_min, args.snr_max, args.n_spaces)
    bers = [[] for _ in range(len(models))]
    fers = [[] for _ in range(len(models))]
    logs = [[] for _ in range(len(models))]
    for snr in snrs:
        dataset = LLRCodeDataset(f"codes/{code_path}", split="test", epoch_len=args.n_samples, EbN0_min=snr, EbN0_max=snr)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.n_samples, num_workers=0)
        for llr, s, x in test_loader:
            llr, s, x = llr.to(device), s.to(device), x.to(device)
            with torch.no_grad():
                for i, m in enumerate(models):
                    llr_ = m(llr, s)
                    x_ = binarize(llr_)
                    fer = FER(x_, x)
                    ber = BER(x_, x)
                    bers[i].append(ber)
                    fers[i].append(fer)
                    if snr in args.eval_points:
                        logs[i].append(-np.log(ber))
                    # print(f"Model: {args.models[i]}, SNR: {snr}, FER: {fer}, BER: {ber}")
    
    c = code.split("_")
    c = c[0] + "(" + c[1][1:] + ", " + c[2][1:] + ")"
    for i, m in enumerate(models):
        plt.plot(snrs, bers[i], label=model_names[i], linestyle='--')
    plt.title(f"BER curve for {c}")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("BER")
    plt.yscale('log')
    plt.legend()
    plt.savefig(f"runs/eval/{c}_BER.png")
    
    plt.clf()
    
    for i, m in enumerate(models):
        plt.plot(snrs, fers[i], label=model_names[i], linestyle='--')
    plt.title(f"FER curve for {c}")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("FER")
    plt.yscale('log')
    plt.legend()
    plt.savefig(f"runs/eval/{c}_FER.png")
    
    print("Eval points at SNR: ", args.eval_points)
    for i, m in enumerate(models):
        print(f"Model: {model_names[i]}")
        print(f"BERS: ", logs[i])
    
    
