import torch

with open("Shakespeare.txt") as file:
    data = file.read()
    

dt = sorted(list(set(data))) # list because later we have to generate tensor by splicing the data and sets cannot be spliced
vocab_size = len(dt)

print(vocab_size)
#Encoding and Decoding
enc = {ch:i for i,ch in enumerate(dt)}
dec = {i:ch for i,ch in enumerate(dt)}
encode = lambda s: [enc[c] for c in s]
decode = lambda t: ''.join([dec[i] for i in t])

#Training and test data
text = torch.tensor(encode(data),dtype=torch.long)
train_split = int(0.9*len(text))
train_data = text[:train_split]
test_data = text[train_split:]
block_size = 8 # deal with 8 charcters at a time while training
batch_size = 4 # Create a tensor of size (4,8) for training at a time

def get_block(s):
    txt = train_data if s.lower()=="train" else test_data
    idx = torch.randint(len(txt)-block_size,(batch_size,)) #generates a tensor containing batch_size number of indices that 
    x = torch.stack([txt[i:i+block_size] for i in idx])
    y = torch.stack([txt[i+1:i+1+block_size] for i in idx])
    return x,y

x1,y1 = get_block("train")
for i in range(batch_size):
    for j in range(block_size):
        #print(decode(x1[i,:j+1].tolist()))
        print(f"Input: {x1[i,:j+1].tolist()}, Target: {y1[i,j].tolist()}")