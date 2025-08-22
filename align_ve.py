import librosa
import numpy as np
from scipy.io import wavfile
from pathlib import Path
import string
from datetime import time
from tempfile import TemporaryDirectory
from gtts import gTTS
from pandas import DataFrame
import pypar
from psola import from_file_to_file
from numba import jit
from time import process_time
from librosa.sequence import dtw


EPS = np.finfo(float).eps


def split_words(text):
    words = text.split(" ")
    for char in string.punctuation:
        if char == "'":
            continue
        words = [word.replace(char, "") for word in words]
    return words


def process_text(text_file, lang, sep, output_path, slow=False):
    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read()
    with open(output_path / "text.txt", "w", encoding="utf-8") as f:
        f.write(text)

    words = split_words(text)
    with open(output_path / "words.txt", "w", encoding="utf-8") as f:
        for w in words:
            f.write(w + "\n")

    if sep != 'Inchangé':
        sep += ' '
        text = sep.join(words)

    text_audio = gTTS(text, lang=lang, slow=slow)
    text_audiofile = output_path / "text.mp3"
    text_audio.save(text_audiofile)

    words_audio = gTTS(". ".join(words), lang=lang, slow=slow)
    words_audiofile = output_path / "words.mp3"
    words_audio.save(words_audiofile)

    return text_audiofile, words_audiofile, words


def extract_stft(audiofile, n_fft, hop_length, sr=None):
    x, sr = librosa.load(audiofile, sr=sr)
    stft_matrix = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))
    t_stft = np.arange(stft_matrix.shape[1]) * hop_length / sr
    return x, sr, stft_matrix, t_stft


def extract_mfcc(audiofile, hop_length, sr=None):
    x, sr = librosa.load(audiofile, sr=sr)
    mfcc_matrix = librosa.feature.mfcc(y=x, sr=sr, hop_length=hop_length)
    return x, sr, mfcc_matrix


def extract_rms(audiofile, hop_length, frame_length, sr=None):
    x, sr = librosa.load(audiofile, sr=sr)
    rms = librosa.feature.rms(y=x, frame_length=frame_length,
                              hop_length=hop_length)[0]
    t = np.arange(rms.size) * hop_length / sr
    return x, sr, rms, t


@jit(nopython=True, cache=True)  # type: ignore
def itakura_saito_divergence(x: np.ndarray, y: np.ndarray) -> float:
    x = x + EPS
    y = y + EPS
    return np.sum((x / y - np.log(x / y) - 1))


def align_animal_text(animal_audiofile, text_audiofile, n_fft, hop_length,
                      representation, output_path=None):
    if representation[0] == 'stft':
        y_text, sr_text, text_representation, _ = extract_stft(
            text_audiofile, n_fft, hop_length)
        y_animal, sr_animal, animal_representation, _ = extract_stft(
            animal_audiofile, n_fft, hop_length, sr=sr_text)
    elif representation[0] == 'rms':
        y_text, sr_text, text_representation, _ = extract_rms(
            text_audiofile, hop_length, frame_length=n_fft)
        y_animal, sr_animal, animal_representation, _ = extract_rms(
            animal_audiofile, hop_length, frame_length=n_fft, sr=sr_text)
    if sr_text != sr_animal:
        raise ValueError(f"Sample rates of text ({sr_text}) " +
                         f"and animal ({sr_animal}) audio files do not match.")

    # _, warping_path = dtw(X=animal_stft, Y=text_stft, backtrack=True,
    #                       metric=itakura_saito_divergence)
    dtw_mat, warping_path = dtw(X=animal_representation, Y=text_representation,
                                backtrack=True)
    warping_path = warping_path[::-1]
    if output_path is not None:
        np.savez(output_path / "animal_text_dtw.npz",
                 dtw_mat=dtw_mat, warping_path=warping_path)
    return warping_path, sr_animal


def align_text_words(text_audiofile, words_audiofile, n_fft, hop_length,
                     representation, output_path=None):
    if representation[0] == 'stft':
        y_text, sr_text, text_representation, _ = extract_stft(text_audiofile,
                                                            n_fft,
                                                            hop_length)
        y_words, sr_words, words_representation, _ = extract_stft(words_audiofile,
                                                               n_fft,
                                                               hop_length)
    elif representation[0] == 'mfcc':
        y_text, sr_text, text_representation = extract_mfcc(
            text_audiofile, hop_length=hop_length)
        y_words, sr_words, words_representation = extract_mfcc(
            words_audiofile, hop_length=hop_length)
    else:
        raise ValueError(f"Unknown representation: {representation[0]}")

    if representation[1] == 'euclidean':
        metric = 'euclidean'
    elif representation[1] == 'itakura_saito':
        metric = itakura_saito_divergence
    else:
        raise ValueError(f"Unknown representation: {representation}")

    if sr_text != sr_words:
        raise ValueError(f"Sample rates of text ({sr_text}) " +
                         f"and words ({sr_words}) audio files do not match.")

    dtw_mat, warping_path = dtw(X=words_representation, Y=text_representation,
                                backtrack=True, metric=metric)
    warping_path = warping_path[::-1]
    if output_path is not None:
        np.savez(output_path / "text_words_dtw.npz",
                 dtw_mat=dtw_mat, warping_path=warping_path)

    words_start_frames, words_end_frames = segment_words(y=y_words,
                                                         hop_length=hop_length)

    text_start_frames = []
    for i in words_start_frames:
        closest = min(warping_path, key=lambda x: abs(x[0] - i))
        text_start_frames.append(closest[1])

    text_end_frames = []
    for i in words_end_frames:
        closest = min(warping_path, key=lambda x: abs(x[0] - i))
        text_end_frames.append(closest[1])

    return text_start_frames, text_end_frames, words_start_frames, \
        words_end_frames, sr_text, sr_words


def segment_words(y, hop_length, frame_length=2048):
    # TODO plot segmentation
    rms = librosa.feature.rms(y=y, frame_length=frame_length,
                              hop_length=hop_length)[0]
    threshold = 0.05 * np.max(rms)
    frames = np.where(rms > threshold)[0]
    min_gap = 50
    words_start_frames = []
    if len(frames) > 0:
        prev = frames[0]
        words_start_frames.append(prev)
        for f in frames[1:]:
            if f - prev > min_gap:
                words_start_frames.append(f)
            prev = f
    words_end_frames = []
    for i in range(1, len(frames)):
        if frames[i] > frames[i - 1] + min_gap:
            words_end_frames.append(frames[i - 1])
    if len(frames) > 0:
        words_end_frames.append(frames[-1])
    return words_start_frames, words_end_frames


def merge_alignments(text_start_frames, text_end_frames,
                     animal_text_warping_path, sr_animal, hop_length, n_fft,
                     words_start_frames, words_end_frames, sr_words, sr_text,
                     words_list, output_path):
    animal_starts = []
    animal_ends = []
    for i in text_start_frames:
        closest_animal = min(animal_text_warping_path,
                             key=lambda x: abs(x[1] - i))
        animal_starts.append(librosa.frames_to_time(closest_animal[0],
                                                    sr=sr_animal,
                                                    hop_length=hop_length,
                                                    n_fft=n_fft))

    for i in text_end_frames:
        closest_animal = min(animal_text_warping_path,
                             key=lambda x: abs(x[1] - i))
        animal_ends.append(librosa.frames_to_time(closest_animal[0],
                                                  sr=sr_animal,
                                                  hop_length=hop_length,
                                                  n_fft=n_fft))

    word_starts = librosa.frames_to_time(words_start_frames, sr=sr_words,
                                         hop_length=hop_length, n_fft=n_fft)
    text_starts = librosa.frames_to_time(text_start_frames, sr=sr_text,
                                         hop_length=hop_length, n_fft=n_fft)
    word_ends = librosa.frames_to_time(words_end_frames, sr=sr_words,
                                       hop_length=hop_length, n_fft=n_fft)
    text_ends = librosa.frames_to_time(text_end_frames, sr=sr_text,
                                       hop_length=hop_length, n_fft=n_fft)

    alignement_times = []
    alignement_seconds = []
    for word, start, end in zip(words_list, animal_starts, animal_ends):
        start_time = seconds2time(start)
        end_time = seconds2time(end)
        alignement_times.append((start_time, end_time, word))
        alignement_seconds.append((start, end, word))

    df = DataFrame(alignement_times, columns=('start', 'end', 'word'))
    df.to_csv(output_path / 'alignment.csv')

    with open(output_path / "alignment.txt", "w", encoding="utf-8") as f:
        for start, end, word in alignement_seconds:
            f.write(f"{start} {end} {word}\n")

    return alignement_seconds, animal_starts, animal_ends, text_starts, \
        text_ends, word_starts, word_ends


def seconds2time(seconds):
    h = int(seconds) // 3600
    seconds -= h * 3600
    m = int(seconds) // 60
    seconds -= m * 60
    s = int(seconds)
    micro = int((seconds - s) * 1e6)
    return time(hour=h, minute=m, second=s, microsecond=micro)


def time2seconds(time):
    h, m, s, micro = time.hour, time.minute, time.second, time.microsecond
    return h * 3600 + m * 60 + s + micro / 1e6


def align(text_file, animal_audiofile, n_fft, hop_length, sep='Inchangé',
          nb_replications=1,
          word_text_representation=None,
          animal_text_representation=None,
          lang='en', output_path="output"):
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)

    text_audiofile, words_audiofile, words = process_text(
        text_file=text_file, lang=lang, output_path=output_path, sep=sep,
        slow=False)

    text_start_frames, text_end_frames, words_start_frames, \
        words_end_frames, sr_text, sr_words = \
        align_text_words(text_audiofile, words_audiofile, n_fft=n_fft,
                         hop_length=hop_length,
                         representation=word_text_representation,
                         output_path=output_path)

    x, sr = librosa.load(animal_audiofile)
    x = np.tile(x, nb_replications)
    # librosa.write_wav(animal_audiofile, x, sr=sr)
    animal_audiofile = output_path / "animal.wav"
    wavfile.write(filename=animal_audiofile, rate=sr, data=x)

    animal_text_warping_path, sr_animal = \
        align_animal_text(animal_audiofile, text_audiofile, n_fft=n_fft,
                          hop_length=hop_length,
                          representation=animal_text_representation,
                          output_path=output_path)

    alignement, animal_starts, animal_ends, text_starts, text_ends, \
        word_starts, word_ends = merge_alignments(
            text_start_frames, text_end_frames, animal_text_warping_path,
            sr_animal, hop_length, n_fft, words_start_frames,
            words_end_frames, sr_words, sr_text, words, output_path)

    if output_path is not None:
        np.savez(output_path / "starts_ends.npz",
                 animal_starts=animal_starts, animal_ends=animal_ends,
                 text_starts=text_starts, text_ends=text_ends,
                 word_starts=word_starts, word_ends=word_ends)
        np.savez(output_path / "params.npz",
                 n_fft=n_fft, hop_length=hop_length, sr=sr)
    # try:
    #     stretch_audio(text_audiofile,
    #                   output_path / "text_aligned_with_animal.mp3",
    #                   words, text_starts, text_ends,
    #                   animal_starts, animal_ends)
    # except ZeroDivisionError:
    #     print("ZeroDivisionError: Skipping stretching of text audio.")

    # try:
    #     stretch_audio(words_audiofile,
    #                   output_path / "words_aligned_with_text.mp3",
    #                   words, word_starts, word_ends, text_starts, text_ends)
    # except ZeroDivisionError:
    #     print("ZeroDivisionError: Skipping stretching of words audio.")

    return alignement


def stretch_audio(audiofile, output_file, words,
                  source_starts, source_ends, target_starts, target_ends):
    def build_pypar_alignment(start_times, end_times, words, alignment_file):
        word_list = []
        current_time = 0.0
        for t_start, t_end, w in zip(start_times, end_times, words):
            # if current_time < t_start:
            silence = pypar.Word(pypar.SILENCE,
                                 [pypar.Phoneme(pypar.SILENCE,
                                                current_time, t_start)])
            word_list.append(silence)
            current_time = t_start

            word = pypar.Word(w, [pypar.Phoneme(w, t_start, t_end)])
            word_list.append(word)
            current_time = t_end - 1e-8
        alignment = pypar.Alignment(word_list)
        alignment.save(alignment_file)

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        source_alignment_file = tmpdir / "source_alignment.json"
        target_alignment_file = tmpdir / "target_alignment.json"
        build_pypar_alignment(source_starts, source_ends, words,
                              source_alignment_file)
        build_pypar_alignment(target_starts, target_ends, words,
                              target_alignment_file)
        from_file_to_file(
            audio_file=audiofile,
            output_file=output_file,
            source_alignment_file=source_alignment_file,
            target_alignment_file=target_alignment_file,
            target_pitch_file=None,
            constant_stretch=None,
        )


if __name__ == "__main__":
    current_path = Path(__file__).parent
    output_path = current_path / "output"
    animal_audiofile = current_path.parent / "data" / "furocontact" / "SampleAnimals" / "Reindeer_sample01.wav"
    animal_audiofile = current_path.parent / "data" / "furocontact" / "SampleAnimals" / "Wolf_Sample01.wav"
    word_text_representation = ('mfcc', 'euclidean')
    # word_text_representation = ('stft', 'euclidean')
    # word_text_representation = ('stft', 'itakura_saito')
    # animal_text_representation = ('stft', 'euclidean')
    animal_text_representation = ('rms', 'euclidean')
    text_file = current_path / "text_sample.txt"
    # text_file = "text_sample_with_dots.txt"
    # text_file = "text_sample_with_commas.txt"
    t0 = process_time()
    alignement_ve = align(
        text_file=text_file,
        animal_audiofile=animal_audiofile,
        n_fft=64,
        hop_length=64,
        lang="fr",
        nb_replications=14,
        word_text_representation=word_text_representation,
        animal_text_representation=animal_text_representation,
        output_path=output_path,
    )
    dt = process_time() - t0
    print(f"Alignment took {dt:.2f} seconds")

    matplotlib_display = False
    if matplotlib_display:
        import display_matplotlib
        display_matplotlib.display_all_figures(path=output_path)
    else:
        import display_plotly
        display_plotly.build_all_figures(path=output_path)
