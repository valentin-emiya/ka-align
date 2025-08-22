from pathlib import Path
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np
import plotly.io as pio
import plotly.express as px
import librosa

import align_ve
import utils


pio.renderers.default = "browser"
margin_bt = 5
margin_lr = 5
margin = {'b': margin_bt, 't': margin_bt, 'l': margin_lr, 'r':margin_lr}


def build_all_figures(path):
    path = Path(path)
    dtw_figures = dict()
    dtw_figures['dtw_animal_text'] = \
        display_dtw(path=path, label1="animal", label2="text" )
    dtw_figures['dtw_text_words'] = \
        display_dtw(path=path, label1="text", label2="words")
    alignment_figures, k_figures, audio_files = plot_alignment_plotly(path)

    grid_alignment_figures = dict()
    for k in alignment_figures:
        grid_alignment_figures[k] = []
        for kk in k_figures:
            audio = utils.load_file_to_contents(audio_files[kk])
            graph = dcc.Graph(figure=alignment_figures[k][kk],
                              config={'displayModeBar': False},
                              id={'type': 'result-graph', 'index': k+kk})
            title1 = dbc.Row(kk, align="right")
            player = html.Audio(src=audio, controls=True,
                                style={"width": "100%"})
            player = dbc.Row(player)
            row = dbc.Row([dbc.Col([title1, player], width=3),
                           dbc.Col(graph, width=9),],
                           align="center")
            grid_alignment_figures[k].append(row)

    return dtw_figures, grid_alignment_figures


def display_dtw(path, label1, label2):
    datafile = path / f"{label1}_{label2}_dtw.npz"
    data = np.load(datafile)
    dtw_mat = data['dtw_mat']
    warping_path = data['warping_path']

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=dtw_mat,
        colorscale='Viridis',
        colorbar=dict(title='DTW cost'),
        showscale=True
    ))

    # Ajout du chemin optimal
    fig.add_trace(go.Scatter(
        x=warping_path[:, 1],
        y=warping_path[:, 0],
        mode='lines',
        line=dict(color='yellow', width=2),
        name='Optimal path'
    ))

    fig.update_xaxes(range=[0, dtw_mat.shape[1] - 1],)
    fig.update_yaxes(range=[0, dtw_mat.shape[0] - 1],)
    fig.update_layout(
        # title=f'DTW - {label1} vs {label2}',
        xaxis_title=f'{label2} (frames)',
        yaxis_title=f'{label1} (frames)',
        margin=margin,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    # fig.write_image(str(datafile.parent / f"dtw_matrix_{label1}_{label2}.pdf"))
    return fig


def display_words(words, starts, ends, fig, row=None, col=None, font_color='black'):
    colors = px.colors.qualitative.Plotly
    for i, (x0 ,x1, w) in enumerate(zip(starts, ends, words)):
        c = colors[i % len(colors)]
        fig.add_vrect(x0=x0, x1=x1,
                      annotation_text=w, annotation_position="top left",
                      annotation_font_color=font_color,
                      line_width=0, fillcolor=c, opacity=0.2,
                      row=row, col=col)


def plot_alignment_plotly(path):
    """
    Variante Plotly de plot_alignment : affiche les trois formes d'onde (animal, text, words)
    et les connexions de début/fin entre elles.
    """
    from plotly.subplots import make_subplots

    figures = dict()
    
    data = np.load(path / "starts_ends.npz")
    animal_starts = data['animal_starts']
    animal_ends = data['animal_ends']
    text_starts = data['text_starts']
    text_ends = data['text_ends']
    word_starts = data['word_starts']
    word_ends = data['word_ends']

    data = np.load(path / "params.npz")
    n_fft = int(data['n_fft'])
    hop_length = int(data['hop_length'])
    sr = int(data['sr'])

    with open(path / "words.txt", "r", encoding="utf-8") as f:
        words = f.readlines()
    words = [w.strip() for w in words if w.strip()]

    figures_keys = ['Animal', 'Text aligned with Animal', 'Text',
                    'Words aligned with Text', 'Words']
    audio_files = {
        'Animal': "animal.wav",
        'Text': "text.mp3",
        'Text aligned with Animal': "text_aligned_with_animal.mp3",
        'Words': "words.mp3",
        'Words aligned with Text': "words_aligned_with_text.mp3",
    }
    for k in audio_files:
        audio_files[k] = path / audio_files[k]
    starts = {
        'Animal': animal_starts,
        'Text': text_starts,
        'Text aligned with Animal': animal_starts,
        'Words': word_starts,
        'Words aligned with Text': text_starts,
    }
    ends = {
        'Animal': animal_ends,
        'Text': text_ends,
        'Text aligned with Animal': animal_ends,
        'Words': word_ends,
        'Words aligned with Text': text_ends,
    }

    stft_colorscale, stft_font_color = 'Brwnyl', 'black'
    stft_colorscale, stft_font_color = 'Viridis', 'white'
    figures['alignment_signal'] = dict()
    figures['alignment_spectro'] = dict()
    figures['alignment_rms'] = dict()
    t_max = dict()
    for k in audio_files:
        y, sr, stft, t_stft = \
            align_ve.extract_stft(audio_files[k], n_fft, hop_length, sr=sr)
        t_y = np.arange(len(y)) / sr
        _, _, y_rms, t_rms = align_ve.extract_rms(
            audio_files[k], hop_length, frame_length=n_fft, sr=sr)
        if 'aligned' in k:
            k_sub = k.split()
            t_max[k] = t_max[k_sub[-1]]
        else:
            t_max[k] = {'y': t_y[-1], 'stft': t_stft[-1], 'rms': t_rms[-1]}

        fig = make_subplots(rows=1, cols=1, shared_xaxes=False)
        fig.add_trace(go.Scatter(x=t_y, y=y, name=k))
        display_words(words=words, starts=starts[k], ends=ends[k], fig=fig)
        fig.update_xaxes(range=[t_y[0], t_max[k]['y']])
        fig.update_layout(showlegend=False, margin=margin)
        figures['alignment_signal'][k] = fig
        
        fig = make_subplots(rows=1, cols=1, shared_xaxes=False)
        y_axis = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        fig.add_trace(go.Heatmap(z=librosa.amplitude_to_db(stft, ref=np.max),
                                x=t_stft, y=y_axis,
                                colorscale=stft_colorscale))
        display_words(words=words, starts=starts[k], ends=ends[k], fig=fig,
                      font_color=stft_font_color)
        fig.update_xaxes(range=[t_stft[0], t_max[k]['stft']])
        fig.update_layout(showlegend=False, margin=margin)
        figures['alignment_spectro'][k] = fig

        fig = make_subplots(rows=1, cols=1, shared_xaxes=False)
        fig.add_trace(go.Scatter(x=t_rms, y=y_rms, name=k))
        display_words(words=words, starts=starts[k], ends=ends[k], fig=fig)
        fig.update_xaxes(range=[t_rms[0], t_max[k]['rms']])
        fig.update_layout(showlegend=False, margin=margin)
        figures['alignment_rms'][k] = fig

    return figures, figures_keys, audio_files

if __name__ == "__main__":
    current_path = Path(__file__).parent
    dtw_figures, grid_alignment_figures = build_all_figures(current_path / "output")
    graph_list = []
    for k in dtw_figures:
        dtw_figures[k].show()
    for k in grid_alignment_figures:
        graph_list += grid_alignment_figures[k]

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    app.layout = dbc.Container(graph_list)
    app.run(debug=True, port=1234)