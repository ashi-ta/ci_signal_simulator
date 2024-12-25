#!/usr/bin/env python3
# encoding: utf-8
"""CI pulse simulator"""

import argparse
import os
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

try:
    from kaldiio import WriteHelper

    use_kaldiio = True
except ImportError:
    use_kaldiio = False


from ci_sim.ci_simulator import CISimulator

parser = argparse.ArgumentParser(description="Simulation of a cochlear implant signal")
# i/o related
parser.add_argument(
    "--wav-list",
    type=str,
    required=True,
    help="WAV list file with paths to WAV files enumerated",
)
parser.add_argument("--csv-list", type=str, help="CSV list file with paths to CSV files enumerated")
parser.add_argument("--outdir", type=str, required=True, help="Output directory")

# operation related
parser.add_argument(
    "--mode",
    type=str,
    required=True,
    choices=["visualize", "data_prep_for_f0", "simulation_only"],
    help="Operation mode",
)
parser.add_argument(
    "--output-all",
    action="store_true",
    help="Output all of data. "
    + "Default limits the data to be output to the F0 range of the output layer of the DNN model being trained",
)

# ci signal related
parser.add_argument(
    "--upsample-sr",
    default=0,
    type=int,
    help="Upsampling frequency to generate a CI signal with high pulses per second (upsample-sr=0, do nothing)",
)
parser.add_argument(
    "--n-band",
    default=8,
    type=int,
    help="Number of n bandwidths when selecting n bandwidths from all m bandwidths",
)
parser.add_argument(
    "--m-band",
    default=8,
    type=int,
    help="Number of m bandwidths when selecting n bandwidths from all m bandwidths",
)
parser.add_argument(
    "--nofm-dur",
    type=float,
    default=10,
    help="Frame length to compute n-of-m analysis in milliseconds",
)
parser.add_argument("--pps", default=800, type=int, help="Pulses per second")
parser.add_argument(
    "--pulse-type",
    default="gaussian",
    type=str,
    choices=["binary", "gaussian"],
    help="Type of pulse signal. 'binary' means biphasic binary pulse.",
)
parser.add_argument(
    "--pulse-size",
    default=0.25,
    type=float,
    help="Number of pulse sample for biphasic binary pulse, or Duration of pulse in millisecond for gaussian pulse",
)
parser.add_argument(
    "--vocoded-signal",
    default="pulse",
    type=str,
    choices=["pulse", "noise", "uniform"],
    help="Signal type for vocoding",
)
parser.add_argument("--freqband-limit", default="0_8000", type=str, help="Frequency bands to be used")
parser.add_argument(
    "--compression-method",
    default="log",
    type=str,
    choices=["log", "powerlaw", "no-compression"],
    help="Algorithm for compressing the dynamic range of the amplitude envelope",
)
parser.add_argument(
    "--freqband-scale-method",
    default="log",
    type=str,
    choices=["octave", "erb", "user", "mel", "log", "linearlog"],
    help="Frequency scale for configuring the filterbank of bandpass filters",
)
parser.add_argument(
    "--env-freq",
    default=300,
    type=float,
    help="Frequency of the low-pass filter when extracting the amplitude envelope. If using 0, not use a lowpass filter.",
)
parser.add_argument(
    "--log-band-freq",
    default=250,
    type=float,
    help="If a log scale is specified for the filter bank (i.e., --freqband-scale-method='log' or 'linearlog'), the starting frequency for that log scale",
)
parser.add_argument(
    "--linearlog-band-linear-num-bands",
    default=7,
    type=int,
    help="If --freqband-scale-method='linearlog', the number of frequency bands within that linear scale",
)
parser.add_argument(
    "--up-downsample-target",
    default="pulse_only",
    type=str,
    choices=["pulse_only", "input_output"],
    help="Manipulation target of upsampling & downsampling. "
    + "If 'pulse_only', apply upsampling & downsampling to pulse signal only. "
    + "If 'input_output', aaply upsampling to input signal & downsampling to output signal only.",
)

# visualization related
parser.add_argument(
    "--detail-vis-start",
    default=0.0,
    type=float,
    help="Start time of the temporal axis when visualizing in detail",
)
parser.add_argument(
    "--detail-vis-end",
    default=0.0,
    type=float,
    help="End time of the temporal axis when visualizing in detail",
)


args = parser.parse_args()
print(args)


class WavReader(object):
    def __init__(self, wav_list):
        self.initialized = False
        self.closed = False
        if wav_list is None:
            raise ValueError("must be specified wav_list")
        else:
            self.file = open(wav_list, "r")
        self.initialized = True

    def __iter__(self):
        with self.file as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                try:
                    k = os.path.splitext(os.path.basename(line))[0]
                    v, sr = librosa.load(line, sr=None)
                except Exception:
                    raise
                yield k, (sr, v)
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.closed:
            self.close()

    def close(self):
        if self.initialized and not self.closed:
            self.file.close()
            self.closed = True


def freq_to_cent(freq):
    return 1200 * np.log2(freq / 10.0)


def cent_to_freq(cent):
    return 10 * 2 ** (cent / 1200)


def data_prep_for_f0(
    ftxt,
    writer,
    key,
    csvs,
    freq_range,
    simres,
    sr,
):
    teacher = np.loadtxt(csvs[key], delimiter=",", dtype=np.float64).reshape(-1, 2)
    for i in range(teacher.shape[0]):
        pitch = teacher[i, 1]
        center_t = int(sr * teacher[i, 0])
        indata = simres[:, center_t - 512 : center_t + 512].copy()
        if indata.shape[1] != 1024:
            continue
        elif len(indata[np.nonzero(indata)]) == 0:
            continue
        else:
            if pitch >= freq_range[0] and pitch <= freq_range[-1]:
                cent_true = freq_to_cent(pitch)
                outdata = str(cent_true)
                datakey = key + "_" + str(teacher[i, 0])
                writer(datakey, indata)
                ftxt.write(f"{datakey} {outdata}\n")
            else:
                continue


def analysis(outdir, key, sr, simres, bands, envs, m_band, wavmat, plt_envs, detail_start=0.0, detail_end=0.0):
    outdir.mkdir(parents=True, exist_ok=True)
    t = simres.shape[1]
    filenames = ["output_pulse"]
    if detail_end - detail_start > 0:
        filenames.append("output_pulse_detail")

    for filename in filenames:
        # plot the waveform
        fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [1, 6]}, figsize=(8, 10))
        axs[0].plot(np.arange(t) / sr, wavmat, linewidth=0.3, color="gray")
        axs[0].set_title("(a) raw waveform")
        axs[0].axes.yaxis.set_ticklabels([])

        if filename == "output_pulse_detail":
            # limit the x-axis range between 500 and 1000 ms
            axs[0].set_xlim(detail_start, detail_end)
            axs[0].set_xticks(np.arange(detail_start, detail_end + 0.01, 0.05))

        for i in range(m_band):
            axs[1].plot(np.arange(t) / sr, simres[i, :] + i + 1, linewidth=0.3, color="gray")
            axs[1].plot(
                np.arange(t) / sr,
                envs[i, :] + i + 1,
                linewidth=1.0,
                linestyle="--",
                color="red",
            )
        axs[1].set_title("(b) CI signal")
        axs[1].set_yticks(np.arange(1, m_band + 1))
        axs[1].set_yticklabels(np.arange(1, m_band + 1)[::-1])
        axs[1].set_ylabel("Elecrodes No.")
        axs[1].set_xlabel("Time [s]")

        if filename == "output_pulse_detail":
            axs[1].set_xlim(detail_start, detail_end)
            axs[1].set_xticks(np.arange(detail_start, detail_end + 0.01, 0.05))

        pp = PdfPages(str(outdir / (key + "_" + filename + ".pdf")))
        pp.savefig(bbox_inches="tight")
        pp.close()
        plt.close(fig)

    if plt_envs:
        # plot only envelopes optionally
        plt.figure(figsize=(8, 15), tight_layout=True)
        plt.subplot(m_band + 1, 1, 1)
        plt.plot(np.arange(t) / sr, wavmat, linewidth=0.3, color="gray")
        plt.title("raw waveform")
        plt.xticks(color="None")
        plt.xlim(0.0, 0.5)

        for i in range(m_band):
            plt.subplot(m_band + 1, 1, m_band + 1 - i)
            plt.plot(np.arange(t) / sr, envs[i, :], linewidth=1)
            band = np.round(bands[i], decimals=1)
            plt.title(
                "Electrode " + str(m_band - i) + " (" + str(band[0]) + "Hz - " + str(band[1]) + "Hz)",
            )
            if i != 0:
                plt.xticks([])
            else:
                plt.xlabel("Time [s]")
            plt.xlim(0.0, 0.5)
        pp = PdfPages(str(outdir / (key + "_env.pdf")))
        pp.savefig()
        pp.close()
        plt.close()


def main():
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # get sampling frequency of input data from first file
    sampling_rate = None
    with WavReader(args.wav_list) as reader:
        for key, (sr, orgmat) in reader:
            if sampling_rate is None:
                sampling_rate = sr
            else:
                if sampling_rate != sr:
                    raise ValueError(
                        f"Sampling frequency is not same between input samples ({sampling_rate} != {sr} in {key})"
                    )

    cochlear_implant = CISimulator(
        sr=sampling_rate,
        upsample_sr=args.upsample_sr,
        nofm_dur=args.nofm_dur,
        n_band=args.n_band,
        m_band=args.m_band,
        pulse_type=args.pulse_type,
        pps=args.pps,
        pulse_size=args.pulse_size,
        compression_method=args.compression_method,
        freqband_scale_method=args.freqband_scale_method,
        freqband_limit=args.freqband_limit,
        erb_band_number_limit="3_35",
        erb_band_number_step=1,
        log_band_freq=args.log_band_freq,
        linearlog_band_linear_num_bands=args.linearlog_band_linear_num_bands,
        filter_impulse_response_method="iir",
        filter_order=6,
        filter_fir_window="hann",
        ext_env_method="halfrect",
        ext_env_impulse_response_method="iir",
        ext_env_filter_order=6,
        ext_env_fir_window="hann",
        ext_env_freq=args.env_freq,
        vocoded_signal=args.vocoded_signal,
        up_downsample_target=args.up_downsample_target,
    )

    # simulate CI pulse signal with F0 label in Kaldi format for F0 training
    if args.mode == "data_prep_for_f0":
        if not use_kaldiio:
            raise ValueError("mode data_prep_for_f0 is required kaldiio package")
        csvs = {}
        with open(args.csv_list, "r") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                k = os.path.splitext(os.path.basename(line))[0]
                csvs[k] = line.strip()

        if args.output_all:
            freq_range = None
        else:
            # frequency range of output files will be limited by the output layer of the DNN model in the paper
            # i.e. from C1 to 20cent * 360
            c1_cent = freq_to_cent(32.70)  # C1 = 32.70 Hz
            cent_range = np.arange(360) * 20 + c1_cent
            freq_range = cent_to_freq(cent_range)

        scpfile = outdir / "file.scp"
        arkfile = outdir / "file.ark"
        textfile = outdir / "text"
        with textfile.open("w", encoding="utf-8") as ftxt:
            with WriteHelper("ark,scp:" + str(arkfile) + "," + str(scpfile)) as writer:
                with WavReader(args.wav_list) as reader:
                    for key, (sr, orgmat) in reader:
                        simres = cochlear_implant.simulate(orgmat)
                        envs = cochlear_implant.envelopes()
                        bands = cochlear_implant.frequency_band()
                        data_prep_for_f0(
                            ftxt,
                            writer,
                            key,
                            csvs,
                            freq_range,
                            simres,
                            sr,
                        )

    # only simulate
    elif args.mode == "simulation_only":
        npy_dir = outdir / "npy"
        npy_dir.mkdir(parents=True, exist_ok=True)
        with open(str(outdir / "result_pickle.list"), "w") as list_f:
            with WavReader(args.wav_list) as reader:
                for key, (sr, orgmat) in reader:
                    simres = cochlear_implant.simulate(orgmat)
                    envs = cochlear_implant.envelopes()
                    bands = cochlear_implant.frequency_band()
                    savenpy = npy_dir / f"{key}.npy"
                    savenpy_env = npy_dir / f"{key}_env.npy"
                    list_f.write(f"{str(savenpy)}\n")
                    np.save(savenpy, simres)
                    np.save(savenpy_env, envs)

    # only visualize
    elif args.mode == "visualize":
        with WavReader(args.wav_list) as reader:
            for key, (sr, orgmat) in reader:
                simres = cochlear_implant.simulate(orgmat)
                envs = cochlear_implant.envelopes()
                bands = cochlear_implant.frequency_band()
                analysis(
                    outdir,
                    key,
                    sr,
                    simres,
                    bands,
                    envs,
                    args.m_band,
                    orgmat,
                    False,
                    args.detail_vis_start,
                    args.detail_vis_end,
                )


if __name__ == "__main__":
    main()
