import os
from dataclasses import dataclass
from omegaconf import OmegaConf
import torch
import numpy as np
from ov_piano import HDF5PathManager
from ov_piano.utils import IncrementalHDF5, TorchWavToLogmel, torch_load_resample_audio
from ov_piano.data.PuDoMS import PuDoMS
from ov_piano.data.midi import GeneralMidiParser, MidiToPianoRoll

@dataclass
class ConfDef:
    INPATH: str = os.path.join("data", "PuDoMS1")
    OUTPUT_DIR: str = "data"
    TARGET_SR: int = 16_000
    STFT_WINSIZE: int = 2048
    STFT_HOPSIZE: int = 384
    MELBINS: int = 229
    MEL_FMIN: int = 50
    MEL_FMAX: int = 8_000
    MIDI_SUS_EXTEND: bool = True
    HDF5_CHUNKLEN_SECONDS: float = 8.0
    DEVICE: str = "cpu"
    IGNORE_MEL: bool = False

if __name__ == "__main__":
    CONF = OmegaConf.structured(ConfDef())
    cli_conf = OmegaConf.from_cli()
    CONF = OmegaConf.merge(CONF, cli_conf)
    print("\n\nCONFIGURATION:")
    print(OmegaConf.to_yaml(CONF), end="\n\n\n")

    METACLASS = PuDoMS
    HDF5_CHUNKLEN = round(CONF.HDF5_CHUNKLEN_SECONDS / (CONF.STFT_HOPSIZE / CONF.TARGET_SR))
    ROLL_HEIGHT = 3 + MidiToPianoRoll.NUM_MIDI_VALUES * 2
    MIDI_QUANT_SECS = CONF.STFT_HOPSIZE / CONF.TARGET_SR

    os.makedirs(CONF.OUTPUT_DIR, exist_ok=True)
    if not CONF.IGNORE_MEL:
        HDF5_MEL_OUTPATH = os.path.join(CONF.OUTPUT_DIR, HDF5PathManager.get_mel_hdf5_basename(f"PuDoMS1", CONF.TARGET_SR, CONF.STFT_WINSIZE, CONF.STFT_HOPSIZE, CONF.MELBINS, CONF.MEL_FMIN, CONF.MEL_FMAX))
    HDF5_ROLL_OUTPATH = os.path.join(CONF.OUTPUT_DIR, HDF5PathManager.get_roll_hdf5_basename(f"PuDoMS1", MIDI_QUANT_SECS, MidiToPianoRoll.NUM_MIDI_VALUES, CONF.MIDI_SUS_EXTEND))

    all = METACLASS(CONF.INPATH, splits=PuDoMS.ALL_SPLITS)

    if not CONF.IGNORE_MEL:
        logmel_fn = TorchWavToLogmel(CONF.TARGET_SR, CONF.STFT_WINSIZE, CONF.STFT_HOPSIZE, CONF.MELBINS, CONF.MEL_FMIN, CONF.MEL_FMAX).to(CONF.DEVICE)
    pianoroll_fn = MidiToPianoRoll()

    if not CONF.IGNORE_MEL:
        h5mel = IncrementalHDF5(HDF5_MEL_OUTPATH, CONF.MELBINS, dtype=np.float32, compression="lzf", data_chunk_length=HDF5_CHUNKLEN, metadata_chunk_length=HDF5_CHUNKLEN, err_if_exists=True)
    h5roll = IncrementalHDF5(HDF5_ROLL_OUTPATH, ROLL_HEIGHT, dtype=np.float32, compression="lzf", data_chunk_length=HDF5_CHUNKLEN, metadata_chunk_length=HDF5_CHUNKLEN, err_if_exists=True)

    print("Computing features...")
    if not CONF.IGNORE_MEL:
        print("Logmels stored into", HDF5_MEL_OUTPATH)
    print("Piano rolls stored into", HDF5_ROLL_OUTPATH)
    loop_length = len(all.data)

    for i, (path, meta) in enumerate(all.data, 1):
        basepath = os.path.basename(path)
        midipath_mid = os.path.join(CONF.INPATH, f"{basepath}.mid")
        midipath_midi = os.path.join(CONF.INPATH, f"{basepath}.midi")

        midipath = None
        if os.path.exists(midipath_mid):
            midipath = midipath_mid
        elif os.path.exists(midipath_midi):
            midipath = midipath_midi
        else:
            print(f"MIDI file for {basepath} not found. Skipping...")
            continue  # ADDED: Skip this iteration if MIDI file is not found

        print(f"Processing MIDI file: {midipath}")
        abspath = os.path.join(CONF.INPATH, path)
        metadata = str((basepath, *meta))

        if not CONF.IGNORE_MEL:
            try:
                wave = torch_load_resample_audio(abspath + METACLASS.AUDIO_EXT, CONF.TARGET_SR, mono=True, normalize_wav=True, device=CONF.DEVICE)
                if wave is None:  # CHANGED: Check if wave is None
                    print(f"Audio file {abspath} could not be loaded. Skipping...")
                    continue  # ADDED: Skip this iteration if audio cannot be loaded
                logmel = logmel_fn(wave).to("cpu").numpy()
                h5mel.append(logmel, metadata)
            except Exception as e:  # ADDED: Catch exceptions during audio loading or logmel computation
                print(f"Error processing audio file {abspath}: {e}. Skipping...")
                continue  # ADDED: Skip this iteration if there's an error in audio processing

        try:
            roll = pianoroll_fn(midipath, GeneralMidiParser, quant_secs=MIDI_QUANT_SECS, extend_offsets_sus=CONF.MIDI_SUS_EXTEND, ignore_redundant_keypress=True, ignore_redundant_keylift=True)
            if roll is None:  # CHANGED: Check if roll is None
                print(f"Piano roll for {midipath} could not be generated. Skipping...")
                continue  # ADDED: Skip this iteration if piano roll cannot be generated
            h5roll.append(roll, metadata)
        except Exception as e:  # ADDED: Catch exceptions during piano roll generation
            print(f"Error processing MIDI file {midipath}: {e}. Skipping...")
            continue  # ADDED: Skip this iteration if there's an error in MIDI processing

        if (i % 5) == 0:
            print(f"[{i}/{loop_length}] {abspath}")

    if not CONF.IGNORE_MEL:
        h5mel.close()
    h5roll.close()
    print("Done!")
