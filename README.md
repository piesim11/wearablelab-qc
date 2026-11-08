# EMG Gesture Recognition

I worked on a pipeline for recognizing hand gestures from muscle activity recordings. I started by inspecting the signals and comparing them across gestures, then prepared the data and trained a neural network.

I also built an interactive viewer to inspect the recorded signals, the model’s internal activations, and its predictions. Moving through the recording makes it easier to compare predicted gestures with the recorded labels and identify mistakes.

## Dataset

The project uses the first recording from subject 01 of the publicly available [UCI EMG Data for Gestures dataset](https://archive.ics.uci.edu/dataset/481/emg+data).

The code expects the recording at:

```text
data/raw/EMG_data_for_gestures-master/01/1_raw_data_13-12_22.03.16.txt
```

The dataset is provided by Krilova et al. under the CC BY 4.0 license.

## Running the project

The following commands run from the project root, where `src/` and `requirements.txt` are located.

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Train the model:

```bash
python -m src.train
```

Training saves the model and its normalization statistics in `emg_model.pt`.

Generate the interactive visualization:

```bash
python -m src.visualize
```

This creates `emg_explorer.html`. When the trained model is already available, the visualization can be generated without repeating the training.

## Viewing the results

On a local computer, `emg_explorer.html` can be opened directly in a browser.

In Codespaces, the following commands serve the visualization:

```bash
mkdir -p docs
cp emg_explorer.html docs/index.html
python -m http.server 8000 --directory docs
```

The page can then be opened through port **8000** in the VS Code **Ports** panel.

The viewer includes:

- The eight recorded EMG signals.
- A heatmap of the model’s filter activations.
- Estimated probabilities for each gesture.
- Recorded and predicted gesture labels.

The slider selects a signal window. Play and Pause control playback of the recorded examples.

## Current scope

This prototype uses a single recording. The reported accuracy is measured on the training data; performance on independent recordings remains to be evaluated.