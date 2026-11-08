import torch

from .dataset import make_windows
from .model import EMGCNN
from .signal_io import load_recording


if __name__ == "__main__":
    # Caricamento dei dati
    data = load_recording()
    X, y = make_windows(data)

    # Normalizzazione per canale
    mean = X.mean(axis=(0, 2), keepdims=True)
    std = X.std(axis=(0, 2), keepdims=True)
    X = (X - mean) / (std + 1e-8)

    X = torch.from_numpy(X)
    y = torch.from_numpy(y)

    # Creazione della rete
    model = EMGCNN()
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training
    for epoch in range(100):
        optimizer.zero_grad()

        scores = model(X)
        loss = loss_fn(scores, y)

        loss.backward()
        optimizer.step()

        print(f"Epoca {epoch + 1}: loss={loss.item():.4f}")

    # Previsioni sui dati usati nel training
    model.eval()

    with torch.no_grad():
        predictions = model(X).argmax(dim=1)

    accuracy = (predictions == y).float().mean().item()
    print(f"Accuratezza training: {accuracy:.1%}")

    print("Gesti corretti: ", (y[:10] + 1).tolist())
    print("Gesti previsti: ", (predictions[:10] + 1).tolist())

    # Salvataggio del modello e della normalizzazione
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mean": torch.from_numpy(mean),
            "std": torch.from_numpy(std),
        },
        "emg_model.pt",
    )

    print("Modello salvato in emg_model.pt")