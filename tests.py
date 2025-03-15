if __name__ == "__main__":
  from coding import *
  
  H = load_pc_matrix("codes/POLAR_N128_K64.txt")
  G = generator_from_nullspace(H)
  assert G.shape == (128, 64), "ERROR: G isn't the right shape"
  assert np.all((H @ G) % 2 == np.zeros((64, 64))), "ERROR: G isn't the null space of H"
  
  x = torch.randint(0, 2, size=(64,), dtype=int)
  x = torch.tensor(G, dtype=int) @ x % 2
  y = bpsk(x)
  x_ = binarize(torch.tensor(y, dtype=torch.float32))
  assert torch.all(x_ == x), "ERROR: binarize isn't working"
  
  dataset = LLRCodeDataset("codes/POLAR_N128_K64.txt", split="train", epoch_len=100, EbN0_min=0, EbN0_max=3)
  print(dataset[0])
  assert np.all(H @ dataset[0][2].numpy() % 2 == np.zeros((64,))), "ERROR: H isn't a valid parity check matrix?"
  
  dataset = LLRCodeDataset("codes/POLAR_N128_K64.txt", split="test", epoch_len=100, EbN0_min=0, EbN0_max=3)
  print(dataset[0])
  assert np.all(H @ dataset[0][2].numpy() % 2 == np.zeros((64,))), "ERROR: H isn't a valid parity check matrix"
  
  