
from audiomentations import Compose, AddGaussianNoise, TimeStretch, \
        PitchShift, Shift, Gain, TanhDistortion, SevenBandParametricEQ, \
        SevenBandParametricEQ, TimeMask

from audiomentations.core.transforms_interface import BaseWaveformTransform


from utils.parselmouth_utils import PitchShiftParselmouth
import random
import librosa
import numpy as np
from random import random
from scipy.signal import lfilter

def aug_factory(augmentation):
    augmentations = []
  
    if augmentation.get("random_phase_mangle", 0):
        augmentations.append(RandomPhaseMangle(min_f=20, max_f=2000, amp=0.99, 
                        p=augmentation["random_phase_mangle"]))
    if augmentation.get("gaussian_noise", 0):
        augmentations.append(AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.05, 
                            p=augmentation["gaussian_noise"]))

    if augmentation.get("time_stretch", 0):
        augmentations.append(TimeStretch(min_rate=0.8, max_rate=1.2, 
                            p=augmentation["time_stretch"]))

    if augmentation.get("pitch_shift_naive", 0):
        #augmentations.append(PitchShift(min_semitones=-3, max_semitones=3, 
                            #p=augmentation["time_stretch"]))
        n_steps = random.choice((-4, 4, 3, -3))
        ps = lambda x, sample_rate=44100: np.cast['float32'](librosa.effects.pitch_shift(x, sr=sample_rate, n_steps=n_steps))
        augmentations.append(ps)

    # Todo formant shift
    # if augmentation.get("formant_shift_parselmouth", 0):

    #     formant_shift_ratio = random.uniform(1, augmentation["formant_shift_parselmouth"])
    #     use_reciprocal = random.uniform(-1, 1) > 0
    #     if use_reciprocal:
    #         formant_shift_ratio = 1 / formant_shift_ratio

    #     ps = lambda x, sample_rate=44100: apply_formant_and_pitch_shift(
    #                                             wav_to_Sound(x, sampling_frequency=sample_rate),
    #                                             formant_shift_ratio=formant_shift_ratio,
    #                                             duration_factor=1.
    #                                         )
    #     augmentations.append(ps)

    if augmentation.get("pitch_shift_parselmouth_prob", 0):

        if augmentation.get("pitch_shift_parselmouth", 0):
            pitch_shift_ratio = augmentation["pitch_shift_parselmouth"]
        else:
            pitch_shift_ratio = 1

        if augmentation.get("pitch_range_parselmouth", 0):
            pitch_range_ratio = augmentation["pitch_range_parselmouth"]
        else:
            pitch_range_ratio = 1
    
        augmentations.append(PitchShiftParselmouth(pitch_shift_ratio, pitch_range_ratio, p=augmentation["pitch_shift_parselmouth_prob"]))

    if augmentation.get("shift", 0):
        augmentations.append(Shift(min_fraction=-0.05, max_fraction=0.2, 
                            p=augmentation["shift"], rollover=True, fade=True))
    
    if augmentation.get("gain", 0):
        augmentations.append(Gain(min_gain_in_db=-6, max_gain_in_db=0, 
                            p=augmentation["gain"]))
    
    if augmentation.get("parametric_eq", 0):
        augmentations.append(SevenBandParametricEQ(min_gain_db = -2, max_gain_db=1, 
                            p=augmentation["parametric_eq"]))
    
    if augmentation.get("tanh_distortion", 0):
        augmentations.append(TanhDistortion(min_distortion=0.1, max_distortion=0.2, 
                            p=augmentation["tanh_distortion"]))
    
    if augmentation.get("time_mask", 0):
        augmentations.append(TimeMask(max_band_part=1/8, 
                            p=augmentation["time_mask"]))
    
    if augmentation.get("reverb", 0):
        #augmentations.append()
        pass
    return augmentations

def aug(signal, 
        augmentations_dict, 
        override=False):
    # Augment the signal

    sig_aug = signal
    # Is its not false it is a dict containing manual transforms to apply
    if override is not False:
        for transform in override.values():
            sig_aug = transform(sig_aug)
        return sig_aug
    if augmentations_dict == False:
        return signal
    transforms = aug_factory(augmentations_dict)
     
    for transform in transforms:
        #print(transform, sig_aug.dtype)
        sig_aug = transform(sig_aug, sample_rate=44100)
    #sig_aug = augment(samples=signal, sample_rate=44100)
    return sig_aug


# From Rave's core
def random_angle(min_f=20, max_f=8000, sr=24000):
    min_f = np.log(min_f)
    max_f = np.log(max_f)
    rand = np.exp(random() * (max_f - min_f) + min_f)
    rand = 2 * np.pi * rand / sr
    return rand

def pole_to_z_filter(omega, amplitude=.9):
    z0 = amplitude * np.exp(1j * omega)
    a = [1, -2 * np.real(z0), abs(z0)**2]
    b = [abs(z0)**2, -2 * np.real(z0), 1]
    return b, a


def random_phase_mangle(x, min_f, max_f, amp, sr):
    angle = random_angle(min_f, max_f, sr)
    b, a = pole_to_z_filter(angle, amp)
    return lfilter(b, a, x)
    

class RandomPhaseMangle(BaseWaveformTransform):
    """Random phase mangle as performed on RAVE"""

    def __init__(self, min_f=20, max_f=2000, amp=0.99, sr=None, p=0.5):
        super().__init__(p)

        self.min_f = min_f
        self.max_f = max_f
        self.amp = amp
        self.sr = sr


    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)


    def apply(self, samples, sample_rate):
        
        if self.sr == None:
            self.sr = sample_rate
        # samples = np.cast['float32'](samples)
        audio = random_phase_mangle(samples, self.min_f, self.max_f, self.amp, self.sr)
            
        return np.squeeze(np.cast['float32'](audio))