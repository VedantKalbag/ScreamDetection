from screamdataset import ScreamDataset
import torch 
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from cnn import CNNNetwork
import plotly.express as px

ANNOTATIONS_FILE = '/home/vedant/projects/ScreamDetection/resources/dataset/dataset-pytorch.csv'
AUDIO_DIR = '/home/vedant/projects/ScreamDetection/resources/dataset/blocked_audio'
BATCH_SIZE = 1024
SAMPLE_RATE=44100
EPOCHS = 30
LEARNING_RATE = 0.001

mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = 44100,
        n_fft = 1024,
        hop_length = 512,
        n_mels = 64
    )
if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

def create_data_loader(train_data,batch_size):
    train_dataloader = DataLoader(train_data,batch_size=batch_size)
    return train_dataloader

def train_one_epoch(model, data_loader,loss_function,optimiser,device):
    for inputs,targets in data_loader:
        inputs,targets = inputs.to(device),targets.to(device)

        # Calculate Loss
        predictions = model(inputs)
        loss = loss_function(predictions,targets)

        # Backpropagate Loss, update weights
        optimiser.zero_grad()
        loss.backward() # Apply backpropagation
        optimiser.step() # Update weights
    print(f"Loss : {loss.item()}")
    return loss.item()

def train(model, data_loader,loss_function, optimiser, device, epochs):
    losses=[]
    epoch=[]
    for i in range(epochs):
        epoch.append(i)
        print(f"Epoch {i+1}:")
        loss = train_one_epoch(model, data_loader, loss_function, optimiser, device)
        losses.append(loss)
        print("-------------------------------------------------------")
    print("Training done")
    return losses, epoch


if __name__ == '__main__':
    #import sys
    #sys.setrecursionlimit(10000)
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'

    print(f"Using device: {DEVICE}")

    

    #instantiating dataset object and transform
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE,
        n_fft = 1024,
        hop_length = 512,
        n_mels = 64
    )

    sd = ScreamDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, DEVICE)
    train_dataloader = create_data_loader(sd,BATCH_SIZE)

    cnn = CNNNetwork().to(DEVICE)
    # Instantiating loss function and optimiser
    loss_function = nn.CrossEntropyLoss()
    optimiser=torch.optim.Adam(cnn.parameters(),
                            lr=LEARNING_RATE
                                )

    #Train Model
    losses,epoch = train(cnn,train_dataloader, loss_function, optimiser, DEVICE, EPOCHS)

    #Save results
    torch.save(cnn.state_dict(),"/home/vedant/projects/ScreamDetection/CNN/trained_models/scream_cnn_crossentropy_adam.pth")

    print("Model trained and stored at /CNN/trained_models/scream_cnn_crossentropy_adam.pth")
    fig = px.line(x=epoch,y=losses)
    fig.show()