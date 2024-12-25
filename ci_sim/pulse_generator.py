#!/usr/bin/env python3
# encoding: utf-8
"""Generate pulsatile signal"""

import numpy as np
from scipy import signal


class PulseGenerator(object):
    def __init__(
        self,
        pulse_type,
        sr,
        upsample_sr,
        dur_sample,
        num_channel,
        pulse_size,
        pps,
        interchannel_sample=None,
    ):
        self.pulse_type = pulse_type
        self.sr = sr
        self.upsample_sr = upsample_sr
        if self.upsample_sr != 0:
            self.sr = self.upsample_sr
        self.dur_sample = dur_sample
        self.num_channel = num_channel
        if self.pulse_type == "gaussian":
            self.pulse_size = int(pulse_size * self.sr / 1000)
            self.std = (self.pulse_size - 1) / 5  # default alpha configuration of gaussian in MATLAB
        else:
            self.pulse_size = int(pulse_size)
        self.pps = pps
        self.interchannel_sample = interchannel_sample

    def __call__(self) -> np.ndarray:
        num_period = int(self.sr / self.pps)
        inter_pulse = num_period - self.pulse_size

        if self.interchannel_sample == 0:
            # NOT interleaved
            interchannel_samples = np.zeros(self.num_channel)
        else:
            if self.interchannel_sample is None:
                # equally space
                self.interchannel_sample = int(num_period / self.num_channel)

            if self.interchannel_sample * self.num_channel > num_period:
                raise ValueError(
                    "interchannel_sample * num_channel={} must be greater than num_period={}".format(
                        self.interchannel_sample * self.num_channel, num_period
                    )
                )

            if self.interchannel_sample < self.pulse_size and self.pulse_type != "gaussian":
                # gaussian pulse is allowed to be overlapped between channels
                raise ValueError(
                    "interchannel_sample={} must be larger than pulse_size={}".format(
                        self.interchannel_sample, self.pulse_size
                    )
                )

            interchannel_samples = np.arange(0, self.interchannel_sample * self.num_channel, self.interchannel_sample)

        if self.pulse_type == "gaussian":
            onepulse = np.concatenate(
                [
                    signal.windows.gaussian(self.pulse_size, std=self.std),
                    np.zeros(inter_pulse),
                ]
            )
        elif self.pulse_type == "binary":
            onepulse = np.concatenate(
                [
                    np.where(np.linspace(1, -1, self.pulse_size) < 0, -1, 1),
                    np.zeros(inter_pulse),
                ]
            )
        else:
            raise ValueError("Invalid pulse_type")

        num_repeat = int(self.dur_sample / num_period) + 1
        pulses = np.tile(onepulse, [self.num_channel, num_repeat])[:, : self.dur_sample]
        channel_pulses = np.array([np.roll(row, x) for row, x in zip(pulses, interchannel_samples)])
        return channel_pulses  # (M, T)


if __name__ == "__main__":
    # use only for test
    import librosa
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    num_channel = 10
    upsample_sr = 64000
    pps = 1200
    pulse = PulseGenerator(
        pulse_type="gaussian",
        sr=16000,
        upsample_sr=upsample_sr,
        dur_sample=upsample_sr,
        num_channel=num_channel,
        pulse_size=0.25,
        pps=pps,
        interchannel_sample=None,
    )()

    fig, axs = plt.subplots(num_channel, 1, sharex=True, figsize=(6, 6))
    fig.subplots_adjust(hspace=0)
    fig.supylabel("Elecrodes No.")

    t = np.arange(0, pulse.shape[1]) / upsample_sr * 1000  # millisecond
    for i in range(pulse.shape[0]):
        axs[i].plot(t, pulse[i, :], linewidth=0.5, color="cornflowerblue")
        axs[i].plot(t, pulse[i, :], "o", markersize=0.5, color="cornflowerblue")
        axs[i].axes.yaxis.set_ticklabels([])
        axs[i].grid(True)
        axs[i].set_xlim(0, 5)
        axs[i].set_yticks([0.0])
        axs[i].set_ylim(-1.1, 1.1)
        axs[i].set_ylabel(str(i + 1))
    axs[i].set_xlabel("Time [ms]")

    pp = PdfPages("biphasic_pulse_sample.pdf")
    pp.savefig(bbox_inches="tight")
    pp.close()
    plt.close()

    pulse = librosa.util.normalize(librosa.resample(pulse, orig_sr=upsample_sr, target_sr=16000), axis=1)
    fig, axs = plt.subplots(num_channel, 1, sharex=True, figsize=(6, 6))
    fig.subplots_adjust(hspace=0)
    fig.supylabel("Elecrodes No.")

    t = np.arange(0, pulse.shape[1]) / 16000 * 1000  # millisecond
    for i in range(pulse.shape[0]):
        axs[i].plot(t, pulse[i, :], linewidth=0.5, color="cornflowerblue")
        axs[i].plot(t, pulse[i, :], "o", markersize=0.5, color="cornflowerblue")
        axs[i].axes.yaxis.set_ticklabels([])
        axs[i].grid(True)
        axs[i].set_xlim(0, 5)
        axs[i].set_yticks([0.0])
        axs[i].set_ylim(-1.1, 1.1)
        axs[i].set_ylabel(str(i + 1))
    axs[i].set_xlabel("Time [ms]")

    pp = PdfPages("biphasic_pulse_sample_downsampling.pdf")
    pp.savefig(bbox_inches="tight")
    pp.close()
    plt.close()
