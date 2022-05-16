
import torch

def sim_cos(enc_x, enc_y, temp=1):
    try:
        return cosim(enc_x, enc_y) / temp
    except NameError:
        cosim = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        return cosim(enc_x, enc_y) / temp

def similarity(enc_x, enc_y, temp=1):
    #return distance_dot(enc_x, enc_y, temp)
    return sim_cos(enc_x, enc_y, temp)

def roll(x):
    return torch.cat((x[-1:], x[:-1]))

def get_audio_length(fn):
    # Returns length of audio file in samples
    import soundfile as sf
    with sf.SoundFile(fn, 'r') as f:
        return f.frames