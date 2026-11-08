from pathlib import Path

import pandas as pd


CHANNELS = [f"channel{i}" for i in range(1, 9)]

DEFAULT_PATH = Path(
    "data/raw/EMG_data_for_gestures-master/01/"
    "1_raw_data_13-12_22.03.16.txt"
)


def load_recording(path=DEFAULT_PATH):
    return pd.read_csv(path, sep="\t")


if __name__ == "__main__":
    from .signal_processing import calculate_rms
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    data = load_recording()

    print(data.head())
    print(data.shape)

    # Durata della registrazione
    N = data.shape[0]
    t_inizio = data["time"].iloc[0]
    t_fine = data["time"].iloc[-1]
    T = (t_fine - t_inizio) / 1000

    print("Numero di campioni:", N)
    print("Primo timestamp (ms):", t_inizio)
    print("Ultimo timestamp (ms):", t_fine)
    print("Durata (s):", T)

    # Intervalli tra campioni consecutivi
    delta_t = data["time"].diff()
    print("Primi intervalli in ms:\n", delta_t.head(40))
    print("Conteggio intervalli:\n", delta_t.value_counts().sort_index())

    # Confronto sullo stesso canale
    riposo = data.loc[data["class"] == 1, "channel1"]
    pugno = data.loc[data["class"] == 2, "channel1"]

    rms_riposo = calculate_rms(riposo)
    rms_pugno = calculate_rms(pugno)

    print("Campioni a riposo:", len(riposo))
    print("Campioni pugno:", len(pugno))
    print("RMS riposo:", rms_riposo)
    print("RMS pugno:", rms_pugno)
    print("Rapporto pugno/riposo:", rms_pugno / rms_riposo)

    # Tempo trascorso dall'inizio, espresso in secondi
    data["time_s"] = (data["time"] - t_inizio) / 1000
    finestra = data.loc[data["time_s"] <= 12]

    print(finestra["class"].value_counts().sort_index())

    graf1 = make_subplots(rows=2, cols=1, shared_xaxes=True)

    graf1.add_trace(
        go.Scatter(
            x=finestra["time_s"],
            y=finestra["channel1"],
            mode="lines",
            name="EMG canale 1",
        ),
        row=1,
        col=1,
    )

    graf1.add_trace(
        go.Scatter(
            x=finestra["time_s"],
            y=finestra["class"],
            mode="lines",
            name="Movimento",
            line_shape="hv",
        ),
        row=2,
        col=1,
    )

    graf1.update_yaxes(title_text="Ampiezza EMG", row=1, col=1)
    graf1.update_yaxes(title_text="Classe", row=2, col=1)
    graf1.update_xaxes(title_text="Tempo (s)", row=2, col=1)
    graf1.update_layout(height=650)

    graf1.show()