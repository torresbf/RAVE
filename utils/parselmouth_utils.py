import parselmouth
import numpy as np
import math
from audiomentations.core.transforms_interface import BaseWaveformTransform
import random
import warnings

PRAAT_CHANGEGENDER_PITCHMEDIAN_DEFAULT = 0.0
PRAAT_CHANGEGENDER_FORMANTSHIFTRATIO_DEFAULT = 1.0
PRAAT_CHANGEGENDER_PITCHSHIFTRATIO_DEFAULT = 1.0
PRAAT_CHANGEGENDER_PITCHRANGERATIO_DEFAULT = 1.0
PRAAT_CHANGEGENDER_DURATIONFACTOR_DEFAULT = 1.0


def wav_to_Sound(wav, sampling_frequency: int = 44100) -> parselmouth.Sound:
    r""" load wav file to parselmouth Sound file
    # __init__(self: parselmouth.Sound, other: parselmouth.Sound) -> None \
    # __init__(self: parselmouth.Sound, values: numpy.ndarray[numpy.float64], sampling_frequency: Positive[float] = 44100.0, start_time: float = 0.0) -> None \
    # __init__(self: parselmouth.Sound, file_path: str) -> None
    returns:
        sound: parselmouth.Sound
    """
    if isinstance(wav, parselmouth.Sound):
        sound = wav
    elif isinstance(wav, np.ndarray):
        sound = parselmouth.Sound(wav, sampling_frequency=sampling_frequency)
    elif isinstance(wav, list):
        wav_np = np.asarray(wav)
        sound = parselmouth.Sound(np.asarray(wav_np), sampling_frequency=sampling_frequency)
    else:
        raise NotImplementedError
    return sound


def get_pitch_median(wav, sr: int = None):
    sound = wav_to_Sound(wav, sr)
    pitch = None
    pitch_median = PRAAT_CHANGEGENDER_PITCHMEDIAN_DEFAULT

    try:
        pitch = parselmouth.praat.call(sound, "To Pitch", 0.8 / 75, 75, 600)
        pitch_median = parselmouth.praat.call(pitch, "Get quantile", 0.0, 0.0, 0.5, "Hertz")
    except Exception as e:
        raise e
        pass

    return pitch, pitch_median

def change_gender(
        sound, pitch=None,
        formant_shift_ratio: float = PRAAT_CHANGEGENDER_FORMANTSHIFTRATIO_DEFAULT,
        new_pitch_median: float = PRAAT_CHANGEGENDER_PITCHMEDIAN_DEFAULT,
        pitch_range_ratio: float = PRAAT_CHANGEGENDER_PITCHRANGERATIO_DEFAULT,
        duration_factor: float = PRAAT_CHANGEGENDER_DURATIONFACTOR_DEFAULT, ) -> parselmouth.Sound:
    try:
        if pitch is None:
            new_sound = parselmouth.praat.call(
                sound, "Change gender", 75, 600,
                formant_shift_ratio,
                new_pitch_median,
                pitch_range_ratio,
                duration_factor
            )
        else:
            new_sound = parselmouth.praat.call(
                (sound, pitch), "Change gender",
                formant_shift_ratio,
                new_pitch_median,
                pitch_range_ratio,
                duration_factor
            )
    except Exception as e:
        raise e

    return new_sound

def apply_formant_and_pitch_shift(
        sound: parselmouth.Sound,
        formant_shift_ratio: float = PRAAT_CHANGEGENDER_FORMANTSHIFTRATIO_DEFAULT,
        pitch_shift_ratio: float = PRAAT_CHANGEGENDER_PITCHSHIFTRATIO_DEFAULT,
        pitch_range_ratio: float = PRAAT_CHANGEGENDER_PITCHRANGERATIO_DEFAULT,
        duration_factor: float = PRAAT_CHANGEGENDER_DURATIONFACTOR_DEFAULT) -> parselmouth.Sound:
    r"""uses praat 'Change Gender' backend to manipulate pitch and formant
        'Change Gender' function: praat -> Sound Object -> Convert -> Change Gender
        see Help of Praat for more details
        # https://github.com/YannickJadoul/Parselmouth/issues/25#issuecomment-608632887 might help
    """

    # pitch = sound.to_pitch()
    pitch = None
    new_pitch_median = PRAAT_CHANGEGENDER_PITCHMEDIAN_DEFAULT
    if pitch_shift_ratio != 1.:
        try:
            pitch, pitch_median = get_pitch_median(sound, None)
            new_pitch_median = pitch_median * pitch_shift_ratio

            # https://github.com/praat/praat/issues/1926#issuecomment-974909408
            pitch_minimum = parselmouth.praat.call(pitch, "Get minimum", 0.0, 0.0, "Hertz", "Parabolic")
            newMedian = pitch_median * pitch_shift_ratio
            scaledMinimum = pitch_minimum * pitch_shift_ratio
            resultingMinimum = newMedian + (scaledMinimum - newMedian) * pitch_range_ratio
            if resultingMinimum < 0:
                new_pitch_median = PRAAT_CHANGEGENDER_PITCHMEDIAN_DEFAULT
                pitch_range_ratio = PRAAT_CHANGEGENDER_PITCHRANGERATIO_DEFAULT

            if math.isnan(new_pitch_median):
                new_pitch_median = PRAAT_CHANGEGENDER_PITCHMEDIAN_DEFAULT
                pitch_range_ratio = PRAAT_CHANGEGENDER_PITCHRANGERATIO_DEFAULT

        except Exception as e:
            raise e

    new_sound = change_gender(
        sound, pitch,
        formant_shift_ratio, new_pitch_median,
        pitch_range_ratio, duration_factor)

    return new_sound

def semitones_to_ratio(x):
    return 2 ** (x / 12)


class PitchShiftParselmouth(BaseWaveformTransform):
    """Pitch shift the sound up or down without changing the tempo"""

    def __init__(self, pitch_ratio=1.4, range_ratio=1.3, p=0.5):
        super().__init__(p)
        # assert min_semitones >= -12
        # assert max_semitones <= 12
        # assert min_semitones <= max_semitones
        # self.min_semitones = min_semitones
        # self.max_semitones = max_semitones

        self.pitch_ratio = pitch_ratio
        self.range_ratio = range_ratio

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["pitch_shift_ratio"] = random.uniform(
                1, self.pitch_ratio)
            
            use_reciprocal = random.uniform(-1, 1) > 0
            if use_reciprocal:
                self.parameters["pitch_shift_ratio"] = 1 / self.parameters["pitch_shift_ratio"]

            self.parameters["pitch_range_ratio"] = random.uniform(
                1, self.range_ratio)

            use_reciprocal = random.uniform(-1, 1) > 0
            if use_reciprocal:
                self.parameters["pitch_range_ratio"] = 1 / self.parameters["pitch_range_ratio"]


    def apply(self, samples, sample_rate):
        
       # samples = np.cast['float32'](samples)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                action='ignore',
                category=parselmouth.PraatWarning,
                message='This application uses RandomPool, which is BROKEN in older releases')
        
            warnings.simplefilter("ignore")
            pitch_shifted_samples = apply_formant_and_pitch_shift(
                                        wav_to_Sound(samples, sampling_frequency=sample_rate),
                                        pitch_shift_ratio=self.parameters["pitch_shift_ratio"],
                                        pitch_range_ratio=self.parameters["pitch_range_ratio"],
                                        duration_factor=1.
                                    )
        return np.squeeze(np.cast['float32'](pitch_shifted_samples.values))