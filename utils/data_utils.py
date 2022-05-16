import soundfile as sf
from tqdm import tqdm
import numpy as np
import os
from utils.core import get_audio_length



def normalize_signal(mix):
    max_abs = np.maximum(np.abs(np.min(mix)), np.max(mix))
    if max_abs > 0:
        mix /= max_abs
    return mix
    
def get_fragment_from_file(fn, nr_samples, normalize=False, from_=0, draw_random=False):
    sample = None
    
    with sf.SoundFile(fn, 'r') as f:
        if nr_samples < 0:
            nr_samples = f.frames
        if draw_random:
            draw_interval_size = np.maximum(int(f.frames - nr_samples), 1)
            from_ = np.random.randint(0, draw_interval_size)
        #assert f.samplerate == 44100, f"sample rate is {f.samplerate}, should be 44100 though."
        if f.samplerate != 44100:
            print(f"Warning: sample rate is {f.samplerate}: {fn}")
        try:
            nr_samples_ = np.minimum(nr_samples, f.frames)
            f.seek(from_)
            sample = f.read(nr_samples_)
            if len(sample.shape) > 1 and sample.shape[1] == 2:
                # to mono
                sample = sample.mean(1)

            # sample too short -> pad
            if len(sample) < nr_samples:
                sample_ = np.zeros((nr_samples,))
                sample_[:len(sample)] = sample
                sample = sample_

            if normalize:
                sample = normalize_signal(sample)
        except Exception as e:
            pass
    return sample

# todo redo func, group by artist
def prepare_fn_groups_vocal(root_folder, 
                        groups=None, 
                        select_only_groups=None,
                        filter_fun_level1=None,
                        group_name_is_folder=True,
                        group_by_artist=False):

    if filter_fun_level1 == None:
        filter_fun_level1 = lambda x: True

    if groups == None:
        groups = {}

    if not group_by_artist:
        fn_counter = 0
        print('Not grouping data by subdir')
    else:
        print('Grouping data by subdir')

    if select_only_groups is not None:
        print(f'Warning: select_only_groups is not None, selecting data only in subdirs {select_only_groups}')
    result = []
    #groups = {}
    for root0, dirs0, _ in tqdm(os.walk(root_folder), desc=f"scanning sub-directories of {root_folder}"):
        for dir0 in dirs0:
            if group_name_is_folder:
                group_name = dir0  # Folder
            else:
                group_name = 'unknown'
                if select_only_groups is not None:
                    print("Warning: group_name_is_folder is False and select_only_groups is not None")

            if select_only_groups is None or \
              (select_only_groups is not None and group_name in select_only_groups):

                fns_level1 = []
                for root1, dirs1, files1 in os.walk(os.path.join(root0, dir0)):
                    for file1 in files1:

                        fn = os.path.join(root1, file1)
                        if filter_fun_level1(fn):
                            if group_by_artist:
                                if group_name in groups:
                                    groups[group_name].append(fn)
                                else:
                                    groups[group_name] = [fn]
                            else:
                                groups[fn_counter] = [fn]    
                                fn_counter += 1
    return groups


def filter1_voice_wav(fn):
    if (fn.endswith("wav") or fn.endswith(
        "WAV")) and ".json" not in fn:
        try:
            if get_audio_length(fn) < (44100 / 10):
                print(f"too short: {fn}")
                return False
        except RuntimeError:
            print(f"exception: {fn}")
            return False
    #print(f"accept {fn}")
        return True
    return False
