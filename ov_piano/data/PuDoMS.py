#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
This module is analogous to ``maps``, but adapted for the PuDoMS dataset.
"""


import os
#
import pandas as pd
#
from .maps import MelMaps, MelMapsChunks


# ##############################################################################
# #  META (paths etc)
# ##############################################################################
class MONO_pudoms:
    """
    This class parses the filesystem tree for the PuDoMS dataset and, based on
    the given filters, stores a list of file paths.

    It can be used to manage PuDoMS files and to create custom dataloaders.
    """

    CSV_NAME = "monophonic_midi.csv"
    ALL_SPLITS = {"train", "validation", "test"}
    AUDIO_EXT = ".wav"
    MIDI_EXT = ".midi"

    def __init__(self, rootpath, splits=None):
        """
        """
        self.rootpath = rootpath
        self.meta_path = os.path.join(rootpath, self.CSV_NAME)
        # filter sanity check
        if splits is None:
            splits = self.ALL_SPLITS
        assert (s in self.ALL_SPLITS for s in splits), \
            f"Unknown split in {splits}"
        # load and filter csv
        df = pd.read_csv(self.meta_path)
        df = df[df["Split"].isin(splits)]
        # reformat into DATA_COLUMNS + metadata_str and gather
        columns = ["File_Number", "Split", "Duration",
                   "Composer", "Title"]
        self.data = []
        for i, (id, s, dur, comp, title) in df[columns].iterrows():
            meta = (s, dur, comp, title)  
            self.data.append((str(id), meta))
        self.full_data = df

    def get_file_abspath(self, basename):
        """
        :param basename: Base name of the corresponding MIDI file without
         extension, e.g.
         MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_08_R1_2015_wav--4'
        :returns: Unique absolute path for that basename
        """
        matches = [fn for fn, _ in self.data if basename in fn]
        assert len(matches) == 1, "Expected exactly 1 match!"
        path = os.path.join(self.rootpath, matches[0])
        return path


# ##############################################################################
# #  PYTORCH DATASETS
# ##############################################################################
class MelPuDoMS(MelMaps):
    """
    Identical to parent class
    """
    pass


class MelPuDoMSChunks(MelMapsChunks):
    """
    Identical to parent class
    """
    pass
