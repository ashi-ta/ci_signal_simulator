#!/usr/bin/env python3
# encoding: utf-8

from typing import Sequence, Tuple

import librosa
import numpy as np

from .pulse_generator import PulseGenerator
from .extract_envelope import extract_envelope
from .filter_signal import filter_signal

OCTAVE_CENTER = [16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000]
UPSAMPLE_TOLERANCE = 5


class CISimulator(object):
    def __init__(
        self,
        sr,
        upsample_sr,
        nofm_dur,
        n_band,
        m_band,
        pulse_type,
        pps,
        pulse_size,
        compression_method,
        freqband_scale_method,
        freqband_limit="0_8000",
        erb_band_number_limit="3_35",
        erb_band_number_step=1,
        log_band_freq=1000,
        linearlog_band_linear_num_bands=7,
        filter_impulse_response_method="fir",
        filter_order=512,
        filter_fir_window="hann",
        ext_env_method="rect",
        ext_env_impulse_response_method="fir",
        ext_env_filter_order=512,
        ext_env_fir_window="hann",
        ext_env_freq=16,
        vocoded_signal="pulse",
        up_downsample_target="pulse_only",
    ):
        self.sr = sr
        if upsample_sr == 0:
            self.upsample_sr = self.sr
        else:
            self.upsample_sr = upsample_sr
        self.upsampling = self.sr != self.upsample_sr
        self.nofm_dur = nofm_dur
        self.n_band = n_band
        self.m_band = m_band
        self.pulse_type = pulse_type
        self.pps = pps
        self.compression_method = compression_method
        self.freqband_scale_method = freqband_scale_method
        self.pulse_size = pulse_size
        self.freqband_limit = freqband_limit
        self.erb_band_number_limit = erb_band_number_limit
        self.erb_band_number_step = erb_band_number_step
        self.log_band_freq = log_band_freq
        self.linearlog_band_linear_num_bands = linearlog_band_linear_num_bands
        self.filter_impulse_response_method = filter_impulse_response_method
        self.filter_order = filter_order
        self.filter_fir_window = filter_fir_window
        self.ext_env_method = ext_env_method
        self.ext_env_impulse_response_method = ext_env_impulse_response_method
        self.ext_env_filter_order = ext_env_filter_order
        self.ext_env_fir_window = ext_env_fir_window
        self.ext_env_freq = ext_env_freq
        self.vocoded_signal = vocoded_signal
        self.up_downsample_target = up_downsample_target

        if self.upsampling and self.vocoded_signal in ["noise", "uniform"]:
            raise ValueError("Cannot use upsampling manipulation & noise or uniform vocoder simultaneously")

        self._configure_frequency_band()

    def simulate(self, orgmat):
        # upsample for generating a CI signal with high pps
        if self.upsampling:
            upsample_mat = librosa.resample(orgmat, orig_sr=self.sr, target_sr=self.upsample_sr)
        else:
            upsample_mat = orgmat.copy()
        upsample_t = upsample_mat.shape[0]
        if self.up_downsample_target == "input_output":
            # use upsampling signal to calculate n-of-m processing
            target_mat = upsample_mat
            nofm_dur = int(self.nofm_dur / 1000 * self.upsample_sr)
        elif self.up_downsample_target == "pulse_only":
            # upsample only pulse signal (multiply envelope and pulse after downsampling)
            target_mat = orgmat.copy()
            nofm_dur = int(self.nofm_dur / 1000 * self.sr)
        target_t = target_mat.shape[0]
        nofm = self.n_band < self.m_band

        # generate upsampled vocode signal
        if self.vocoded_signal == "pulse":
            pulse = PulseGenerator(
                pulse_type=self.pulse_type,
                sr=self.sr,
                upsample_sr=self.upsample_sr,
                dur_sample=upsample_t,
                num_channel=self.n_band,
                pulse_size=self.pulse_size,
                pps=self.pps,
                interchannel_sample=None,
            )()  # (N, T)
            # when vocoding, multiply the pulse signal from the basal first (highest freqency first)
            pulse = np.flip(pulse, axis=0)
            # downsampling signal
            if self.upsampling and self.up_downsample_target == "pulse_only":
                pulse = librosa.util.normalize(
                    librosa.resample(pulse, orig_sr=self.upsample_sr, target_sr=self.sr), axis=1
                )
                if pulse.shape[1] < target_t:
                    pulse = np.concatenate([pulse, np.zeros(target_t - pulse.shape[1])], 1)
                elif pulse.shape[1] > target_t:
                    pulse = pulse[:, :target_t]
        elif self.vocoded_signal == "noise":
            pulse = np.random.normal(loc=0, scale=1, size=[self.n_band, upsample_t]).astype(np.float32)
        elif self.vocoded_signal == "uniform":
            pulse = np.ones([self.n_band, upsample_t], dtype=np.float32)
        else:
            raise ValueError("Invalid vocoded_signal")
        # calculate the number of frames and padding for the n-of-m frame processing
        if nofm:
            total_frames = 1 + int(target_t // nofm_dur)
            target_mat = np.concatenate([target_mat, np.zeros(nofm_dur)])
            pulse = np.concatenate([pulse, np.zeros([self.n_band, nofm_dur])], axis=1)
            pad_t = target_mat.shape[0]
        else:
            pad_t = target_t

        # Pre-emphasis
        # https://librosa.org/doc/main/generated/librosa.effects.preemphasis.html
        outsig = librosa.effects.preemphasis(target_mat, coef=0.97)

        band_envs = np.empty(pad_t)
        for band in self.bands:
            x_cloned = outsig.copy()
            if band[0] == 0 and band[1] >= self.sr / 2:
                pass  # do nothing when freq_band=[0, nyquist_freq]
            else:
                if band[0] == 0:
                    btype = "lowpass"
                    freq_array = band[1]
                elif band[1] >= self.sr / 2:
                    if band[0] >= self.sr / 2:
                        raise ValueError("Invalid bandwidth = (" + str(band[0]) + ", " + str(band[1]) + ").")
                    btype = "highpass"
                    freq_array = band[0]
                else:
                    btype = "bandpass"
                    freq_array = band

                x_cloned = filter_signal(
                    x_cloned,
                    btype,
                    freq_array,
                    self.filter_impulse_response_method,
                    self.filter_order,
                    self.filter_fir_window,
                    self.sr,
                )
            env = extract_envelope(
                x_cloned,
                self.ext_env_method,
                self.ext_env_freq,
                self.ext_env_impulse_response_method,
                self.ext_env_filter_order,
                self.ext_env_fir_window,
                self.sr,
            )
            band_envs = np.vstack([band_envs, env])  # (M+1, T)
        band_envs = band_envs[1:, :]  # (M, T)

        # N-of-M selection -> (M, T)
        if nofm:
            nband_envs = np.zeros([self.m_band, pad_t])
            nofm_pulse = np.zeros([self.m_band, pad_t])
            for j in range(total_frames):
                frame_index_low = j * self.nofm_dur
                frame_index_upp = (j + 1) * self.nofm_dur
                frame_envs = band_envs[:, frame_index_low:frame_index_upp]
                top_n_index = np.sort(
                    np.argpartition(np.sum(np.abs(frame_envs), axis=-1), -self.n_band)[-self.n_band :]
                )
                nband_envs[top_n_index, frame_index_low:frame_index_upp] = frame_envs[top_n_index, :]
                nofm_pulse[top_n_index, frame_index_low:frame_index_upp] = pulse[:, frame_index_low:frame_index_upp]
            nband_envs = nband_envs[:, :target_t]
            nofm_pulse = nofm_pulse[:, :target_t]
        else:
            nband_envs = band_envs
            nofm_pulse = pulse

        # Compression -> (M, T)
        if self.compression_method == "log":
            steep = 200
            if np.any(nband_envs > 1):
                print("appear the sample > 1")
            nband_envs = np.clip(nband_envs, 0, 1)
            nband_envs = np.log(1 + steep * nband_envs) / np.log(1 + steep)
        elif self.compression_method == "powerlaw":
            steep = 0.3
            nband_envs = np.power(nband_envs, steep)
        else:
            pass

        self.envs = nband_envs.copy()
        sim_res = nband_envs * nofm_pulse

        if self.upsampling and self.up_downsample_target == "input_output":
            sim_res = librosa.resample(sim_res, orig_sr=self.upsample_sr, target_sr=self.sr)
            self.envs = librosa.resample(self.envs, orig_sr=self.upsample_sr, target_sr=self.sr)
            t_diff = sim_res.shape[1] - orgmat.shape[0]
            assert abs(t_diff) <= UPSAMPLE_TOLERANCE
            if t_diff > 0:
                sim_res = sim_res[:, :-t_diff]
                self.envs = self.envs[:, :-t_diff]
            else:
                sim_res = np.hstack([sim_res, np.zeros([sim_res.shape[0], abs(t_diff)])])
                self.envs = np.hstack([self.envs, np.zeros([self.envs.shape[0], abs(t_diff)])])
        return sim_res

    def _configure_frequency_band(self):
        """Generate frequency band configuration"""

        freqband_limit = np.array(self.freqband_limit.split("_")).astype(np.float64)

        if self.freqband_scale_method == "octave":
            octband, _ = octave_band(freqband_limit[0], freqband_limit[1], self.sr)
            b = np.concatenate([np.zeros(1), octband])  # band start with 0Hz
        elif self.freqband_scale_method == "erb":
            erb_band_number_limit = np.array(self.erb_band_number_limit.split("_")).astype(np.int64)
            erbband = erb_band(
                erb_band_number_limit[0],
                erb_band_number_limit[1],
                freqband_limit[0],
                freqband_limit[1],
                self.sr,
            )
            b = erbband[:: self.erb_band_number_step]
        elif self.freqband_scale_method == "user":
            b = np.array(self.user_freqband.split("_")).astype(np.float64)
        elif self.freqband_scale_method == "mel":
            mel_limit = librosa.hz_to_mel(freqband_limit).astype(np.float64)
            b_mel = np.linspace(mel_limit[0], mel_limit[1], self.m_band + 1)
            b = librosa.mel_to_hz(b_mel)
        elif self.freqband_scale_method == "log":
            if freqband_limit[0] == 0:
                b = np.geomspace(self.log_band_freq, freqband_limit[1], self.m_band)
                b = np.concatenate([np.zeros(1), b])
            else:
                b = np.geomspace(freqband_limit[0], freqband_limit[1], self.m_band + 1)
        elif self.freqband_scale_method == "linearlog":
            b_linear = np.linspace(
                freqband_limit[0],
                self.log_band_freq,
                self.linearlog_band_linear_num_bands,
            )
            b_log = np.geomspace(
                self.log_band_freq,
                freqband_limit[1],
                self.m_band + 2 - self.linearlog_band_linear_num_bands,
            )
            b = np.concatenate((b_linear[:-1], b_log))

        if self.m_band > b.shape[0] - 1:
            raise ValueError(
                "m_band={} must be smaller than the number of bands={}".format(self.m_band, b.shape[0] - 1)
            )

        # divisions of frequency band
        # e.g. lower_freq=500, upper_freq=8000
        #      num_freqband=1: bands=[[0, 4000]]
        #      num_freqband=2: bands=[[0, 500], [500, 4000]]
        #      num_freqband=3: bands=[[0, 500], [500, 1000], [1000, 4000]]
        #      num_freqband=4: bands=[[0, 500], [500, 1000], [1000, 2000], [2000, 4000]]
        self.bands = []
        if self.m_band == 1:
            self.bands.append([freqband_limit[0], freqband_limit[1]])
        else:
            for i in range(self.m_band):
                i += 1
                if i == self.m_band:
                    self.bands.append([b[i - 1], b[-1]])
                else:
                    self.bands.append([b[i - 1], b[i]])
        print("Frequency band = " + ", ".join([str(i[0]) + "_" + str(i[1]) for i in self.bands]))

    def frequency_band(self):
        return self.bands

    def envelopes(self):
        return self.envs


def octave_band(
    lower_freq: float = 500, upper_freq: float = 8000, sr: int = 16000
) -> Tuple[np.ndarray, Sequence[np.ndarray]]:
    """Return center frequency of 1/1 octave band and cutoff frequency of octave filter.
    Center freqency is based on nominal center frequency (ANSI S1.11)

    Args:
        lower_freq: lower limit of frequency. Defaults to 500.
        upper_freq: upper limit of frequency. Defaults to 8000.
        sr: sampling frequency. Defaults to 16000.

    Returns:
        Center frequency of 1/1 octave band and cutoff frequency of octave filter
    """
    if lower_freq < 0 or upper_freq > (sr / 2):
        raise ValueError(
            "Lower and upper frequencies must be "
            "between 0 and Nyquist frequency (sampling frequency / 2), but got lower={} and upper={}".format(
                lower_freq, upper_freq
            )
        )
    octcenter = np.array([i for i in OCTAVE_CENTER if i >= lower_freq and i <= upper_freq])
    octcutoff = [np.array([i / np.sqrt(2), i * np.sqrt(2)]) for i in octcenter]
    return octcenter, octcutoff


def erb_band(
    lower_band: int = 3,
    upper_band: int = 35,
    lower_freq: float = 0,
    upper_freq: float = 8000,
    sr: int = 16000,
) -> np.ndarray:
    """Return Equivalent rectangular bandwidth (ERB).
    This function is based on erb2hz function of MATLAB audio Toolbox.
    (Reference: https://jp.mathworks.com/help/audio/ref/erb2hz.html)

    Args:
        lower_band: lower limit of band number. Defaults to 3.
        upper_band: upper limit of band number. Defaults to 35.
        lower_freq: lower limit of frequency. Defaults to 0.
        upper_freq: upper limit of frequency. Defaults to 8000.
        sr: sampling frequency. Defaults to 16000.

    Returns:
        Equivalent rectangular bandwidth (ERB) limited by each parameters.
    """
    if lower_freq < 0 or upper_freq > (sr / 2):
        raise ValueError(
            "Lower and upper frequencies must be "
            "between 0 and Nyquist frequency (sampling frequency / 2), but got lower={} and upper={}".format(
                lower_freq, upper_freq
            )
        )
    erb = np.arange(lower_band, upper_band + 1)
    a = 1000 * np.log(10) / (24.7 * 4.37)
    f = (10 ** (erb / a) - 1) / 0.00437
    f_limit = np.array([i for i in f if i >= lower_freq and i <= upper_freq])

    if f.shape[0] > f_limit.shape[0]:
        f_del = np.array([i for i in f if i <= lower_freq or i >= upper_freq])
        print(
            "The number of ERB is limited by lower_freq ({}Hz) or upper_freq ({}Hz) from {} to {}.".format(
                lower_freq,
                upper_freq,
                f.shape[0],
                f_limit.shape[0],
            )
            + " The limited frequencies are {}.".format(f_del)
        )

    return f_limit
