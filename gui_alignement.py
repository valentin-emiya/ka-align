import dash
from dash import dcc, html, Output, Input, State
import soundfile as sf
import base64
import os
import tempfile
from pathlib import Path


def read_audio_file(file_path):
    try:
        y, sr = sf.read(file_path)
        print(y.shape, sr)
        return y, sr
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None, None


def read_text_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            s = f.read()
            print(s)
            return s
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""


# Dummy align_text_animal function for demonstration
def align_text_animal(fichier_son, fichier_texte):
    y, sr = read_audio_file(fichier_son)
    s = read_text_file(fichier_texte)
    alignment = []
    for i, w in enumerate(s.split(' ')):
        minutes = i // 60
        seconds = i % 60
        alignment.append((f"{minutes:02}:{seconds:02}", w))
    print(f"Alignement: {alignment}")
    return alignment


app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1("Alignement son d'animal & paroles"),
    html.H2("Son animal"),
    dcc.Upload('Glissez-déposez ou cliquez pour sélectionner un fichier',
               id='upload-son',
               accept='.wav,.mp3',
               style={'width': '100%',
                      'height': '60px',
                      'lineHeight': '60px',
                      'borderWidth': '1px',
                      'borderStyle': 'dashed',
                      'borderRadius': '5px',
                      'textAlign': 'center'}),
    html.H2("Paroles"),
    dcc.Upload('Glissez-déposez ou cliquez pour sélectionner un fichier',
               id='upload-paroles',
               accept='.txt',
               style={'width': '100%',
                      'height': '60px',
                      'lineHeight': '60px',
                      'borderWidth': '1px',
                      'borderStyle': 'dashed',
                      'borderRadius': '5px',
                      'textAlign': 'center'}),
    html.Button("Télécharger l'alignment", id="btn-download"),
    dcc.Download(id="download-alignment"),
])

@app.callback(
    Output('upload-son', 'children'),
    Input('upload-son', 'filename'),
    prevent_initial_call=True,
)
def update_son(filename):
    return filename


@app.callback(
    Output('upload-paroles', 'children'),
    Input('upload-paroles', 'filename'),
    prevent_initial_call=True,
)
def update_paroles(filename):
    return filename


@app.callback(
    Output("download-alignment", "data"),
    Input("btn-download", "n_clicks"),
    State('upload-son', 'contents'),
    State('upload-son', 'filename'),
    State('upload-paroles', 'contents'),
    State('upload-paroles', 'filename'),
    prevent_initial_call=True,
)
def func(n_clicks,
         son_content, son_filename, paroles_content, paroles_filename):
    if not (son_content and paroles_content):
        print(son_content, paroles_content)
        return dict(content="Veuillez uploader les deux fichiers.",
                    filename="error.txt")

    sound_path = save_temp_file(son_content, son_filename)
    paroles_path = save_temp_file(paroles_content, paroles_filename)

    align_list = align_text_animal(fichier_son=sound_path,
                                   fichier_texte=paroles_path)
    s = '\n'.join([f"{a} {b}" for a, b in align_list])
    print(s)
    sfilename = Path(son_filename).stem
    pfilename = Path(paroles_filename).stem
    out_filename = f"alignement__{sfilename}__with__{pfilename}.txt"
    return dict(content=s, filename=out_filename)


def save_temp_file(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    suffix = os.path.splitext(filename)[1]
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, 'wb') as f:
        f.write(decoded)
    return path


if __name__ == '__main__':
    app.run(debug=True)
