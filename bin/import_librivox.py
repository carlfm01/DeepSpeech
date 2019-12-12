#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

# Make sure we can import stuff from util/
# This script needs to be run from the root of the DeepSpeech repository
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import codecs
import fnmatch
import pandas
import progressbar
import subprocess
import tarfile
import unicodedata

from sox import Transformer
from util.downloader import maybe_download
from tensorflow.python.platform import gfile

SAMPLE_RATE = 16000

def _download_and_preprocess_data(data_dir):
    # Conditionally download data to data_dir
    print("Downloading Librivox data set (55GB) into {} if not already present...".format(data_dir))
    with progressbar.ProgressBar(max_value=7, widget=progressbar.AdaptiveETA) as bar:
        TEST_CLEAN_URL = "https://skgpgg.by.files.1drv.com/y4m0_NKS_iRTKRp108JiwWUNFCsOa_5T838pbjXQYMms5Mq3vkHX4CJNtI8atUTEUJxuHaezOgyLVMAaK4Cd03bzoLf76pWH1awYd8IR5P-Fq9MYIWnAOurAiTULlOWzloSfj_8WyLMt9Rhu9iFEfpBevmHBZasvmcQUZAL5WlWizW4md_TErkxa8wvHQAqaj-x/test-clean.tar.gz"
        TEST_OTHER_URL = "http://www.openslr.org/resources/12/test-other.tar.gz"

        def filename_of(x): return os.path.split(x)[1]
        test_clean = maybe_download(filename_of(TEST_CLEAN_URL), data_dir, TEST_CLEAN_URL)
        bar.update(5)
        test_other = maybe_download(filename_of(TEST_OTHER_URL), data_dir, TEST_OTHER_URL)
        bar.update(6)

    # Conditionally extract LibriSpeech data
    # We extract each archive into data_dir, but test for existence in
    # data_dir/LibriSpeech because the archives share that root.
    print("Extracting librivox data if not already extracted...")
    with progressbar.ProgressBar(max_value=7, widget=progressbar.AdaptiveETA) as bar:
        LIBRIVOX_DIR = "LibriSpeech"
        work_dir = os.path.join(data_dir, LIBRIVOX_DIR)
        
        _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "test-clean"), test_clean)
        bar.update(5)
        _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "test-other"), test_other)
        bar.update(6)

    # Convert FLAC data to wav, from:
    #  data_dir/LibriSpeech/split/1/2/1-2-3.flac
    # to:
    #  data_dir/LibriSpeech/split-wav/1-2-3.wav
    #
    # And split LibriSpeech transcriptions, from:
    #  data_dir/LibriSpeech/split/1/2/1-2.trans.txt
    # to:
    #  data_dir/LibriSpeech/split-wav/1-2-0.txt
    #  data_dir/LibriSpeech/split-wav/1-2-1.txt
    #  data_dir/LibriSpeech/split-wav/1-2-2.txt
    #  ...
    print("Converting FLAC to WAV and splitting transcriptions...")
    with progressbar.ProgressBar(max_value=7,  widget=progressbar.AdaptiveETA) as bar:
        
        test_clean = _convert_audio_and_split_sentences(work_dir, "test-clean", "test-clean-wav")
        bar.update(5)
        test_other = _convert_audio_and_split_sentences(work_dir, "test-other", "test-other-wav")
        bar.update(6)

    # Write sets to disk as CSV files
    test_clean.to_csv(os.path.join(data_dir, "librivox-test-clean.csv"), index=False)
    test_other.to_csv(os.path.join(data_dir, "librivox-test-other.csv"), index=False)

def _maybe_extract(data_dir, extracted_data, archive):
    # If data_dir/extracted_data does not exist, extract archive in data_dir
    if not gfile.Exists(os.path.join(data_dir, extracted_data)):
        tar = tarfile.open(archive)
        tar.extractall(data_dir)
        tar.close()

def _convert_audio_and_split_sentences(extracted_dir, data_set, dest_dir):
    source_dir = os.path.join(extracted_dir, data_set)
    target_dir = os.path.join(extracted_dir, dest_dir)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Loop over transcription files and split each one
    #
    # The format for each file 1-2.trans.txt is:
    #  1-2-0 transcription of 1-2-0.flac
    #  1-2-1 transcription of 1-2-1.flac
    #  ...
    #
    # Each file is then split into several files:
    #  1-2-0.txt (contains transcription of 1-2-0.flac)
    #  1-2-1.txt (contains transcription of 1-2-1.flac)
    #  ...
    #
    # We also convert the corresponding FLACs to WAV in the same pass
    files = []
    for root, dirnames, filenames in os.walk(source_dir):
        for filename in fnmatch.filter(filenames, '*.trans.txt'):
            trans_filename = os.path.join(root, filename)
            with codecs.open(trans_filename, "r", "utf-8") as fin:
                for line in fin:
                    # Parse each segment line
                    first_space = line.find(" ")
                    seqid, transcript = line[:first_space], line[first_space+1:]

                    # We need to do the encode-decode dance here because encode
                    # returns a bytes() object on Python 3, and text_to_char_array
                    # expects a string.
                    transcript = unicodedata.normalize("NFKD", transcript)  \
                                            .encode("ascii", "ignore")      \
                                            .decode("ascii", "ignore")

                    transcript = transcript.lower().strip()

                    # Convert corresponding FLAC to a WAV
                    flac_file = os.path.join(root, seqid + ".flac")
                    wav_file = os.path.join(target_dir, seqid + ".wav")
                    if not os.path.exists(wav_file):
                        tfm = Transformer()
                        tfm.set_output_format(rate=SAMPLE_RATE)
                        tfm.build(flac_file, wav_file)
                    wav_filesize = os.path.getsize(wav_file)

                    files.append((os.path.abspath(wav_file), wav_filesize, transcript))

    return pandas.DataFrame(data=files, columns=["wav_filename", "wav_filesize", "transcript"])

if __name__ == "__main__":
    _download_and_preprocess_data(sys.argv[1])
