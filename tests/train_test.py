import torch
from src.cifar_train import Encoder, Decoder, LitAutoEncoder

def test_encoder_output():
    model = Encoder()
    x = torch.randn(2, 3*32*32)  # batch di 2 immagini
    z = model(x)
    print("Encoder output shape:", z.shape)
    # deve restituire 3 numeri per ogni immagine
    assert z.shape == (2, 3)

def test_decoder_output():
    model = Decoder()
    z = torch.randn(2, 3)  # batch di 2 vettori latenti
    x_hat = model(z)
    print("Decoder output shape:", x_hat.shape)
    # deve ricostruire immagine flattenata
    assert x_hat.shape == (2, 3*32*32)

def test_training_step_loss():
    enc = Encoder()
    dec = Decoder()
    autoenc = LitAutoEncoder(enc, dec)

    x = torch.randn(2, 3, 32, 32)  # batch di 2 immagini random
    x_flat = x.view(x.size(0), -1)
    z = autoenc.encoder(x_flat)
    x_hat = autoenc.decoder(z)
    loss = torch.nn.functional.mse_loss(x_hat, x_flat)
    print("Loss calcolata:", loss.item())
    # la loss deve essere un numero positivo
    assert loss.item() >= 0