import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .dataset import make_windows
from .model import EMGCNN
from .signal_io import load_recording


GESTURES = [
    "Riposo",
    "Pugno",
    "Flessione del polso",
    "Estensione del polso",
    "Deviazione radiale",
    "Deviazione ulnare",
]


if __name__ == "__main__":
    # Carica il modello già addestrato.
    saved = torch.load(
        "emg_model.pt",
        map_location="cpu",
        weights_only=True,
    )

    model = EMGCNN()
    model.load_state_dict(saved["model_state_dict"])
    model.eval()

    # Prepara le finestre con la normalizzazione del training.
    raw, labels = make_windows(load_recording())
    X = (torch.from_numpy(raw) - saved["mean"]) / (saved["std"] + 1e-8)

    with torch.no_grad():
        activations = model.features[:4](X).numpy()
        probabilities = model(X).softmax(dim=1).numpy()

    activation_max = float(activations.max())

    def title(i):
        predicted = int(probabilities[i].argmax())
        return (
            "EMG Gesture Explorer<br><sup>"
            f"Finestra {i + 1}/{len(raw)}"
            f" · Annotato: {GESTURES[labels[i]]}"
            f" · Previsto: {GESTURES[predicted]}"
            " · Prima registrazione, dati di training</sup>"
        )

    def traces(i):
        # Otto segnali originali.
        signals = [
            go.Scatter(
                y=raw[i, c],
                mode="lines",
                name=f"Canale {c + 1}",
            )
            for c in range(8)
        ]

        # Risposta dei 64 filtri dopo la seconda convoluzione.
        heatmap = go.Heatmap(
            z=activations[i],
            y=list(range(1, 65)),
            colorscale="Magma",
            zmin=0,
            zmax=activation_max,
            colorbar=dict(title="Attivazione", len=0.28, y=0.53),
        )

        # Probabilità stimate dei sei gesti.
        predicted = int(probabilities[i].argmax())
        bars = go.Bar(
            x=probabilities[i] * 100,
            y=GESTURES,
            orientation="h",
            marker_color=[
                "#65dec9" if c == predicted else "#405773"
                for c in range(6)
            ],
            text=[f"{p:.1%}" for p in probabilities[i]],
            textposition="outside",
            cliponaxis=False,
            showlegend=False,
        )

        return signals + [heatmap, bars]

    # Crea i tre pannelli.
    fig = make_subplots(
        rows=3,
        cols=1,
        vertical_spacing=0.12,
        subplot_titles=(
            "1 · Gli otto segnali EMG",
            "2 · Attivazioni dei 64 filtri",
            "3 · Probabilità stimate dei gesti",
        ),
    )

    initial = traces(0)

    for trace in initial[:8]:
        fig.add_trace(trace, row=1, col=1)

    fig.add_trace(initial[8], row=2, col=1)
    fig.add_trace(initial[9], row=3, col=1)

    # Ogni frame rappresenta una finestra.
    fig.frames = [
        go.Frame(
            name=str(i),
            data=traces(i),
            traces=list(range(10)),
            layout=go.Layout(title_text=title(i)),
        )
        for i in range(len(raw))
    ]

    # Scale fisse per confrontare le finestre.
    for row in (1, 2):
        fig.update_xaxes(
            title_text="Campione nella finestra",
            range=[0, raw.shape[2] - 1],
            row=row,
            col=1,
        )

    fig.update_yaxes(
        title_text="Ampiezza (unità del dataset)",
        range=[float(raw.min()), float(raw.max())],
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Filtro appreso",
        range=[0.5, 64.5],
        row=2,
        col=1,
    )
    fig.update_xaxes(
        title_text="Probabilità stimata (%)",
        range=[0, 110],
        row=3,
        col=1,
    )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=GESTURES,
        autorange="reversed",
        row=3,
        col=1,
    )

    fig.update_layout(
        title_text=title(0),
        template="plotly_dark",
        height=1100,
        paper_bgcolor="#0b1220",
        plot_bgcolor="#111d30",
        margin=dict(t=150, b=180, l=170, r=100),
        legend=dict(orientation="h", y=1.06),

        # Slider per scegliere la finestra.
        sliders=[dict(
            active=0,
            x=0.23,
            len=0.77,
            y=-0.10,
            currentvalue=dict(prefix="Finestra: "),
            steps=[
                dict(
                    label=str(i + 1),
                    method="animate",
                    args=[
                        [str(i)],
                        dict(
                            mode="immediate",
                            frame=dict(duration=0, redraw=True),
                            transition=dict(duration=0),
                        ),
                    ],
                )
                for i in range(len(raw))
            ],
        )],

        # Pulsanti di riproduzione.
        updatemenus=[dict(
            type="buttons",
            direction="left",
            x=0,
            y=-0.10,
            yanchor="top",
            showactive=False,
            bgcolor="#24364c",
            font=dict(color="#eaf1fb"),
            buttons=[
                dict(
                    label="▶ Play",
                    method="animate",
                    args=[
                        None,
                        dict(
                            fromcurrent=True,
                            frame=dict(duration=500, redraw=True),
                            transition=dict(duration=0),
                        ),
                    ],
                ),
                dict(
                    label="⏸ Pausa",
                    method="animate",
                    args=[
                        [None],
                        dict(
                            mode="immediate",
                            frame=dict(duration=0, redraw=False),
                            transition=dict(duration=0),
                        ),
                    ],
                ),
            ],
        )],
    )

    fig.write_html(
        "emg_explorer.html",
        include_plotlyjs=True,
        auto_play=False,
    )

    print("Creato emg_explorer.html: aprilo nel browser.")