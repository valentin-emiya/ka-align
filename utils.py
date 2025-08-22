import base64
from pathlib import Path


def save_contents_to_file(contents, filename):
    out_dir = Path('assets')
    out_dir.mkdir(exist_ok=True)
    print(out_dir, filename)
    filepath = out_dir / filename


    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    with open(filepath, 'wb') as f:
        f.write(decoded)
    return str(filepath)


def load_file_to_contents(filepath):
    """
    Charge un fichier et le convertit en chaîne base64 compatible Dash Upload.
    Par défaut, mime_type="audio/wav" (adapter selon le type de fichier).
    """
    ext = Path(filepath).suffix
    mime_type = {'.mp3': "audio/mp3",
                 '.wav': "audio/wav",
                 '.txt': "text/plain",
                 }
    filepath = Path(filepath)
    with open(filepath, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    contents = f"data:{mime_type[ext]};base64,{encoded}"
    return contents