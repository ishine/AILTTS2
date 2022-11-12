from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from text import text_to_sequence
from utils import pad_1D, pad_2D, process_meta
from preprocessors.utils import get_alignment
import tgt
import librosa
import torch
import random


def prepare_dataloader(data_path, filename, batch_size, segment_size=8192, shuffle=True, split=True, num_workers=2, meta_learning=False, seed=0):
    dataset = TextMelDataset(data_path, filename, segment_size=segment_size, split=split)
    sampler = None
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, shuffle=shuffle, 
                    collate_fn=dataset.collate_fn, drop_last=True, num_workers=num_workers) 
    return loader


def replace_outlier(values, max_v, min_v):
    values = np.where(values<max_v, values, max_v)
    values = np.where(values>min_v, values, min_v)
    return values


def norm_mean_std(x, mean, std):
    x = (x - mean) / std
    return x

class TextMelDataset(Dataset):
    def __init__(self, data_path, filename="train.txt", segment_size=8192, split=True):
        self.data_path = data_path
        self.basename, self.text, self.sid = process_meta(os.path.join(data_path, filename))

        self.sid_dict = self.create_speaker_table(self.sid)

        with open(os.path.join(data_path, 'stats.json')) as f:
            data = f.read()
        stats_config = json.loads(data)
        self.f0_stat = stats_config["f0_stat"] # max, min, mean, std
        self.energy_stat = stats_config["energy_stat"] # max, min, mean, std
        #TODO Need to change manually
        self.f0_stat = [330.58, 0.0, 161.24, 57.34]
        self.energy_stat = [67.17, 0.0, 20.80, 15.56]
        
        self.create_sid_to_index()
        self.segment_size = segment_size
        self.split = split
        print('Speaker Num :{}'.format(len(self.sid_dict)))
        print('Loading waveform with 16kHz sampling rate')
    
    def create_speaker_table(self, sids):
        speaker_ids = np.sort(np.unique(sids))
        d = {speaker_ids[i]: i for i in range(len(speaker_ids))}
        return d

    def create_sid_to_index(self):
        _sid_to_indexes = {} 
        # for keeping instance indexes with the same speaker ids
        for i, sid in enumerate(self.sid):
            if sid in _sid_to_indexes:
                _sid_to_indexes[sid].append(i)
            else:
                _sid_to_indexes[sid] = [i]
        self.sid_to_indexes = _sid_to_indexes

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        sid = self.sid_dict[self.sid[idx]]
        phone = np.array(text_to_sequence(self.text[idx], []))
        mel_path = os.path.join(
            self.data_path, "mel", "libritts-mel-{}.npy".format(basename))
        mel_target = np.load(mel_path)
        #* wav2vec2.0
        latent_path = os.path.join(
            self.data_path, "latent", "libritts-latent-{}.npy".format(basename))
        latent = np.load(latent_path)
        D_path = os.path.join(
            self.data_path, "alignment", "libritts-ali-{}.npy".format(basename))
        D = np.load(D_path)
        f0_path = os.path.join(
            self.data_path, "f0", "libritts-f0-{}.npy".format(basename))
        f0 = np.load(f0_path)
        f0 = replace_outlier(f0,  self.f0_stat[0], self.f0_stat[1])
        f0 = norm_mean_std(f0, self.f0_stat[2], self.f0_stat[3])
        energy_path = os.path.join(
            self.data_path, "energy", "libritts-energy-{}.npy".format(basename))
        energy = np.load(energy_path)
        energy = replace_outlier(energy, self.energy_stat[0], self.energy_stat[1])
        energy = norm_mean_std(energy, self.energy_stat[2], self.energy_stat[3])

        #* wav load (same setting with libritts.py), SR = 16000
        resampled_wav_path = os.path.join(
            self.data_path, 'wav16', str(self.sid[idx]), '{}.wav'.format(basename))

        trimmed_path = resampled_wav_path.replace('.wav','.trimmed_wav.pt')
        pase_len = len(mel_target)
        
        if os.path.exists(trimmed_path):
            audio = torch.load(trimmed_path)
        else:
            tg_path = os.path.join(self.data_path, 'TextGrid', str(self.sid[idx]), '{}.TextGrid'.format(basename))
            textgrid = tgt.io.read_textgrid(tg_path)
            _, _, start, end = get_alignment(textgrid.get_tier_by_name('phones'), 16000, 256)
            audio, _ = librosa.load(resampled_wav_path, sr=None)
            audio = audio[int(16000*start):int(16000*end)]
            audio = torch.FloatTensor(audio)
            audio = audio.unsqueeze(0)
            audio = audio[:,:pase_len*256]
            torch.save(audio, trimmed_path)
        
        audio_start_idx = 0 #* Training -> n / Test -> 0
        
        if self.split:
            if pase_len > self.segment_size//256: #audio.size(1) >= self.segment_size:
                audio = torch.nn.functional.pad(audio, (0, self.segment_size), 'constant')
                max_audio_start_idx = pase_len - self.segment_size//256 ## margin 2
                audio_start_idx = random.randint(0, max_audio_start_idx)
                # print(audio.size(1), audio_start_idx*256, audio_start_idx*256 + self.segment_size)
                audio = audio[:, audio_start_idx*256: audio_start_idx*256 + self.segment_size]
            else:
                audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')
                # print('audio length forced to be :',audio.size(1))
        
        audio = audio.squeeze(0)
        
        sample = {"id": basename,
                "sid": sid,
                "text": phone,
                "mel_target": mel_target,
                "latent": latent,
                "audio": audio,
                "audio_start_idx": audio_start_idx,
                "D": D,
                "f0": f0,
                "energy": energy}
            
        return sample

    def reprocess(self, batch, cut_list):
        ids = [batch[ind]["id"] for ind in cut_list]
        sids = [batch[ind]["sid"] for ind in cut_list]
        audios = [batch[ind]["audio"] for ind in cut_list]
        audio_idxs = [batch[ind]["audio_start_idx"] for ind in cut_list]
        latents = [batch[ind]["latent"] for ind in cut_list]
        texts = [batch[ind]["text"] for ind in cut_list]
        mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
        Ds = [batch[ind]["D"] for ind in cut_list]
        f0s = [batch[ind]["f0"] for ind in cut_list]
        energies = [batch[ind]["energy"] for ind in cut_list]
        for text, D, id_ in zip(texts, Ds, ids):
            if len(text) != len(D):
                print(text, text.shape, D, D.shape, id_)
        length_text = np.array(list())
        for text in texts:
            length_text = np.append(length_text, text.shape[0])

        length_mel = np.array(list())
        for mel in mel_targets:
            length_mel = np.append(length_mel, mel.shape[0])

        length_latent = np.array(list())
        for latent in latents:
            length_latent = np.append(length_latent, latent.shape[0])
        texts = pad_1D(texts)
        Ds = pad_1D(Ds)
        mel_targets = pad_2D(mel_targets)
        latents = pad_2D(latents)
        f0s = pad_1D(f0s)
        energies = pad_1D(energies)
        log_Ds = np.log(Ds + 1.)

        out = {"id": ids,
               "sid": np.array(sids),
               "text": texts,
               "mel_target": mel_targets,
               "latent": latents,
               "audio": audios,
               "audio_start_idx": audio_idxs,
               "D": Ds,
               "log_D": log_Ds,
               "f0": f0s,
               "energy": energies,
               "src_len": length_text,
               "mel_len": length_mel,
               "latent_len": length_latent}
        
        return out

    def collate_fn(self, batch):
        len_arr = np.array([d["text"].shape[0] for d in batch])
        index_arr = np.argsort(-len_arr)
        output = self.reprocess(batch, index_arr)

        return output