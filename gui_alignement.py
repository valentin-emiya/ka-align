# TODO traiter proprement le cas où la figure est vide
# TODO ajouter la lecture des sons
# TODO ajouter les paramètre n_fft et hop
# TODO faire le scénario simple avec l'alignement mots / animal
from pathlib import Path
from time import process_time

import dash
from dash import dcc, html, Output, Input, State, Patch, MATCH, callback
import dash_bootstrap_components as dbc
import librosa
import gtts
from scipy.io import wavfile
import soundfile as sf

import align_ve
import utils
import display_plotly


ALL_LANGS = gtts.lang.tts_langs()
ALL_LANGS_REV = {ALL_LANGS[k]: k for k in ALL_LANGS}

SR_GTTS = 24000

FIGURES = dict()

#######
# Aux #
#######
def align_text_animal(fichier_son, fichier_texte, nb_replications, n_fft,
                      hop_length, sep, lang):
    output_path = "output"
    animal_audiofile = fichier_son
    word_text_representation = ('mfcc', 'euclidean')
    animal_text_representation = ('rms', 'euclidean')
    text_file = fichier_texte
    t0 = process_time()
    alignment = align_ve.align(
        text_file=text_file,
        animal_audiofile=animal_audiofile,
        n_fft=n_fft,
        hop_length=hop_length,
        sep=sep,
        lang=ALL_LANGS_REV[lang],
        nb_replications=nb_replications,
        word_text_representation=word_text_representation,
        animal_text_representation=animal_text_representation,
        output_path=output_path,
    )
    dt = process_time() - t0
    print(f"Alignment took {dt:.2f} seconds")
    alignment = [(start, word) for start, end, word in alignment]
    dtw_figures, grid_alignment_figures = \
        display_plotly.build_all_figures(path=output_path)
    for k in dtw_figures:
        FIGURES[k] = dtw_figures[k]
    for k in grid_alignment_figures:
        FIGURES[k] = grid_alignment_figures[k]
    return alignment, output_path


##############
# Components #
##############
upload_sound = \
    dcc.Upload("Glissez-déposez ou cliquez pour sélectionner un fichier",
               id="upload-son",
               accept=".wav,.mp3",
               style={"width": "100%",
                      "height": "60px",
                      "lineHeight": "60px",
                      "borderWidth": "1px",
                      "borderStyle": "dashed",
                      "borderRadius": "5px",
                      "textAlign": "center",
                      })
label_duration = dbc.Label("", id="label-duration")
label_sr = dbc.Label("", id="label-sr")
upload_paroles = \
    dcc.Upload("Glissez-déposez ou cliquez pour sélectionner un fichier",
               id="upload-paroles",
               accept=".txt",
               style={"width": "100%",
                      "height": "60px",
                      "lineHeight": "60px",
                      "borderWidth": "1px",
                      "borderStyle": "dashed",
                      "borderRadius": "5px",
                      "textAlign": "center",},)
label_nb_words = dbc.Label("", id="label-nb_words")
input_nb_replications = dcc.Input(id="nb_replications", type="number",
                                  value=1, min=1,
                                  style={"width": "100%"}, )
label_nb_replications = dbc.Label("Nombre de répétitions du son",
                                  html_for="nb_replications")
label_lang = dbc.Label("Langue", html_for="lang")
dropdown_lang = dcc.Dropdown(list(ALL_LANGS.values()), 'French',
                             id="lang", style={"width": "100%"}, )
label_n_fft = dbc.Label("Longueur analyse", html_for="lang")
dropdown_n_fft = dcc.Dropdown([2 ** p for p in range(4, 10)], 64,
                              id="n_fft", style={"width": "100%"}, )
label_dur_fft = dbc.Label("", id='n_fft_dur')
label_hop = dbc.Label("Hop", html_for="hop")
dropdown_hop = dcc.Dropdown([2 ** p for p in range(4, 10)], 64,
                            id="hop", style={"width": "100%"}, )
label_dur_hop = dbc.Label("", id='hop_dur')
label_sep = dbc.Label("Modifier séparateur", html_for="sep")
dropdown_sep = dcc.Dropdown(list(['Inchangé', '.', ',', ' ']), 'Inchangé',
                            id="sep", style={"width": "100%"}, )


button_download = dbc.Button("Calculer et télécharger l'alignement",
                             id="btn-download")
input_fig_height = dcc.Input(id="fig_height",type="number",
                             value=800, min=1, style={"width": "100%"}, )
label_fig_height = dbc.Label("Hauteur de la figure", html_for="fig_height")
credits = dbc.Card([dbc.CardBody([
    html.H3("Crédits"),
    html.P(
        "Ce projet a été réalisé en 2025"
        " par Tinhinane Hamoum et Valentin Emiya"
        " au Laboratoire d'Informatique et des Systèmes,"
        " à l'Université d'Aix-Marseille (France),"
        " pour l'installation Karaoké Animal/Furomancy de Damien Beyrouthy."
        )])])

#######
# App #
#######
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dcc.Store(id="animal-audiofile-store"),
    dcc.Store(id="paroles-file-store"),
    html.H1("Alignement son d'animal & paroles"),
    html.H2("Son animal"),
    dbc.Row([dbc.Col(upload_sound, width=8,),
             dbc.Col([dbc.Row(label_duration),
                      dbc.Row(label_sr),], width=4)]),
    dbc.Col(html.Audio(id="animal_player", controls=True, hidden=True), width=8),
    html.H2("Paroles"),
    dbc.Row([dbc.Col(upload_paroles, width=8),
             dbc.Col(label_nb_words, width=4,),]),
    dbc.Button("Paramètres", id="collapse-button", color="secondary"),
    dbc.Collapse([
        dbc.Row([dbc.Col(label_nb_replications, width=3),
                 dbc.Col(input_nb_replications, width=2,),
                ]),
        dbc.Row([dbc.Col(label_lang, width=3),
                 dbc.Col(dropdown_lang, width=2,),]),
        dbc.Row([dbc.Col(label_n_fft, width=3),
                dbc.Col(dropdown_n_fft, width=2,),
                dbc.Col(label_dur_fft, width=2)]),
        dbc.Row([dbc.Col(label_hop, width=3),
                dbc.Col(dropdown_hop, width=2,),
                dbc.Col(label_dur_hop, width=2)]),
        dbc.Row([dbc.Col(label_sep, width=3),
                dbc.Col(dropdown_sep, width=2,),]),
                ], id="collapse-params", is_open=False,),
    html.Br(),
    button_download,
    dcc.Download(id="download-alignment"),
    html.H2("Résultats"),
    # dbc.Row([
    #     html.Audio(id="text_player", controls=True, hidden=False),
    #     html.Audio(id="text_aligned_player", controls=True, hidden=False),
    #     html.Audio(id="words_player", controls=True, hidden=False),
    #     html.Audio(id="words_aligned_player", controls=True, hidden=False),
    #     ]),
    html.Div([
        dcc.Tabs(id='tabs-figures', value='alignment_signal', children=[
            dcc.Tab(label='Alignement signaux', value='alignment_signal'),
            dcc.Tab(label='Alignement spectrogrammes',
                    value='alignment_spectro'),
            dcc.Tab(label='Alignement énergie', value='alignment_rms'),
            dcc.Tab(label='DTW animal/texte', value='dtw_animal_text'),
            dcc.Tab(label='DTW texte/mots', value='dtw_text_words'),
        ]),
        html.Div([], id='graph-results'),
        # dcc.Graph(id='graph-results', figure={'layout': {'height': 10}}),
        dbc.Row([dbc.Col(label_fig_height, width=2),
                 dbc.Col(input_fig_height, width=1,),
                ]),
    ]),
    credits,
    ]
)

#############
# Callbacks #
#############
@callback(
    Output('upload-son', 'children'),
    Output('animal-audiofile-store', 'data'),
    Output('label-duration', 'children'),
    Output('label-sr', 'children'),
    Output('animal_player', 'src'),
    Output('animal_player', 'hidden'),
    # Output('graph0', 'figure'),
    Input('upload-son', 'filename'),
    Input('upload-son', 'contents'),
    prevent_initial_call=True,
)
def update_son(filename, contents):
    audiofilename = utils.save_contents_to_file(contents, filename)
    try:
        y, sr = librosa.load(audiofilename, sr=None)
        duration = y.shape[0] / sr
        duration_str = f"Durée: {duration:.2f}s"
        sr_str = f"Fréquence d'échantillonnage: {sr} Hz"
    except Exception as e:
        duration_str = f"Erreur de lecture: {e}"
        sr_str = ""
        y = None
        sr = None
    audio_contents = utils.load_file_to_contents(audiofilename)
    return filename, audiofilename, duration_str, sr_str, audio_contents, False


@callback(
    Output('upload-paroles', 'children'),
    Output('paroles-file-store', 'data'),
    Output('label-nb_words','children'),
    Input('upload-paroles', 'filename'),
    Input('upload-paroles', 'contents'),
    prevent_initial_call=True,
)
def update_paroles(filename, contents):
    print('update_paroles', filename)
    paroles_filename = utils.save_contents_to_file(contents, filename)

    with open(paroles_filename, "r", encoding="utf-8") as f:
        text = f.read()
    words = align_ve.split_words(text)
    n_words = f"Nombre de mots: {len(words)}"
    return filename, paroles_filename, n_words

@app.callback(
    Output("collapse-params", "is_open"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse-params", "is_open")],
    prevent_initial_call=True,
)
def toggle_collapse(n, is_open):
    return not is_open

@callback(
    Output("download-alignment", "data"),
    Output('tabs-figures', 'value'),
    # Output('text_player', 'src'),
    # Output('words_player', 'src'),
    # Output('text_aligned_player', 'src'),
    # Output('words_aligned_player', 'src'),
    # Output('text_player', 'hidden'),
    # Output('words_player', 'hidden'),
    # Output('text_aligned_player', 'hidden'),
    # Output('words_aligned_player', 'hidden'),
    Input("btn-download", "n_clicks"),
    State('animal-audiofile-store', 'data'),
    State('paroles-file-store', 'data'),
    State('nb_replications', 'value'),
    State('lang', 'value'),
    State('n_fft', 'value'),
    State('hop', 'value'),
    State('sep', 'value'),
    State('tabs-figures', 'value'),
    prevent_initial_call=True,
)
def process_and_download(n_clicks, son_path, paroles_path, nb_replications,
                         lang, n_fft, hop_length, sep, figure_tab):
    if not (son_path and paroles_path):
        return dict(content="Veuillez uploader les deux fichiers.",
                    filename="error.txt")
    son_path = Path(son_path)
    paroles_path = Path(paroles_path)
    align_list, output_path = align_text_animal(fichier_son=son_path,
                                                fichier_texte=paroles_path,
                                                nb_replications=nb_replications,
                                                n_fft=n_fft,
                                                hop_length=hop_length,
                                                sep=sep,
                                                lang=lang)
    s = '\n'.join([f"{a} {b}" for a, b in align_list])
    print(s)
    sfilename = son_path.stem
    pfilename = paroles_path.stem
    out_filename = f"alignement__{sfilename}__with__{pfilename}.txt"
    
    # output_path = Path(output_path)
    # text_contents = utils.load_file_to_contents(output_path / 'text.mp3')
    # words_contents = utils.load_file_to_contents(output_path / 'words.mp3')
    # words_aligned_contents = utils.load_file_to_contents(
    #     output_path / 'words_aligned_with_text.mp3')
    # text_aligned_contents = utils.load_file_to_contents(
    #     output_path / 'text_aligned_with_animal.mp3')
    
    return dict(content=s, filename=out_filename), figure_tab
        # text_contents, words_contents, text_aligned_contents, \
        # words_aligned_contents, False, False, False, False


@callback(
    Output('graph-results', 'children'),
    Output('fig_height', 'value'),
    Input('tabs-figures', 'value'),
    State('fig_height', 'value'),
    prevent_initial_call=True,
)
def show_figure_tab(tab, height):
    if tab not in FIGURES:
        return None, 800
    elif 'dtw' in tab:
        div_children = [dcc.Graph(figure=FIGURES[tab],
                                  id={'type': 'result-graph', 'index': tab})]
        return div_children, height
    else:
        div_children = FIGURES[tab]
        return div_children, height
            #     dcc.Tab(label='Alignement signaux', value='alignment_signal'),
            # dcc.Tab(label='Alignement spectrogrammes',
            #         value='alignment_spectro'),
            # dcc.Tab(label='Alignement énergie', value='alignment_rms'),
            # dcc.Tab(label='DTW animal/texte', value='dtw_animal_text'),
            # dcc.Tab(label='DTW texte/mots', value='dtw_text_words'),

@callback(
    Output('n_fft_dur', 'children'),
    Input('n_fft', 'value'),
)
def updage_n_fft(n_fft):
    n_fft_dur = n_fft / SR_GTTS * 1000
    n_fft_dur = f'{n_fft_dur:.1f} ms (sr = {SR_GTTS/1000} kHz)'
    return n_fft_dur


@callback(
    Output('hop_dur', 'children'),
    Input('hop', 'value'),
)
def updage_hop(hop_length):
    hop_dur = hop_length / SR_GTTS * 1000
    hop_dur = f'{hop_dur:.1f} ms'
    return hop_dur


@callback(
    # Output("graph-results", "figure", allow_duplicate=True),
    Output({'type': 'result-graph', 'index': MATCH},
            'figure', allow_duplicate=True),
    Input('fig_height', 'value'),
    State('tabs-figures', 'value'),
    State({'type': 'result-graph', 'index': MATCH},
           'figure'),
    prevent_initial_call=True,
)
def set_figure_height(height, tab, fig):
    if 'alignment' in tab:
        height = (height + 4 * 30) // 5 
    p = Patch()
    p['layout']['height'] = height
    return p

if __name__ == '__main__':
    app.run(debug=True, dev_tools_hot_reload=False)
