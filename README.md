# Cochlear implant (CI) signal simulator

Basic cochlear implant (CI) signal simulator coded only in Python

## Key Features

- Support Continuous-Interleaved Sampling (CIS) strategy
  - output is an interleaved biphasic pulse modulated by amplitude envelopes corresponding to each frequency band
  - modifiable some parameters such as pulses per second (pps), number of channels, compression methods, and frequency scales of the filterbank
- Support for inserting an n-of-m algorithm into basic CIS (i.e., Advanced Combination Encoder (ACE) strategy)
- Support not only a simple interleaved biphasic pulse (1 or 0) but also an interleaved Gaussian pulse with upsampling
  - when simulating CI signal on digital, the interleaved condition between channels is not always followed due to the limitation of the sampling theorem
  - also, there is a possibility of occurring aliasing when combining the amplitude envelope and pulse signal
  - in that case, use Gaussian pulse with upsampling followed by downsampling to match the sampling frequency of an input signal

## Requirements

- This toolkit depends on the following libraries:
  - [NumPy](https://numpy.org/)
  - [Matplotlib](https://matplotlib.org/)
  - [SciPy](https://scipy.org/)
  - [librosa](https://librosa.org/doc/latest/index.html)
  - (optional) [kaldiio](https://github.com/nttcslab-sp/kaldiio)
    - only use for the data preparation to train the F0 prediction model
    - not required if only simulation or visualization
- You can install them individually or `pip install -r requirements.txt`

## Usage

### CI simulation

- Create a text file containing the path to the sound files (.wav format) as shown below:

```Text
/wav/path/test1.wav
/wav/path/test2.wav
/wav/path/test3.wav
...
```

- As an example, to simulate the 8-channel CIS signal at 600-pps with a simple interleaved biphasic pulse, use the following command

```Shell
# wav.list is the list file prepared in the above procedure

python main.py \
    --mode simulation_only \
    --wav-list wav.list \
    --outdir tmp \
    --n-band 8 \
    --m-band 8 \
    --pps 600 \
    --pulse-type binary \
    --pulse-size 2
```

- The result numpy object will be saved in `tmp/npy` and the list file of the npy paths will be saved in `tmp/result_pickle.list`
- The object file includes numpy array and can be laod like:

```python
# when the output numpy object is "tmp/npy/test.npy"
>>> import numpy as np
>>> a = np.load("tmp/npy/test.npy")
>>> a.shape
(8, 5096)  # (number of channels, duration of input signal)
```

- In the above case, when using an input signal at a sampling rate of 16000, the upper-bound pulse rate is 1000 pps
  - because the pulse-to-pulse interval per channel would require 16 samples (8-ch * 2-sample)
- To simulate the higher pulse rate signal (e.g., 1600-pps with an interleaved Gaussian pulse), use the following command

```Shell
python main.py \
    --mode simulation_only \
    --wav-list wav.list \
    --outdir tmp \
    --n-band 8 \
    --m-band 8 \
    --pps 1600 \
    --upsample-sr 64000
```

- In this case, the interleaved Gaussian pulse is generated at a sampling rate of 64000 and then downsampled to the sampling rate of an input signal

### Visualize

- As an example, to visualize a 20-channel CIS signal at 400-pps, use the following command

```Shell
# wav.list is the list file prepared in the above procedure

python main.py \
    --mode visualize \
    --wav-list wav.list \
    --outdir tmp \
    --n-band 20 \
    --m-band 20 \
    --pps 400 \
    --pulse-type binary \
    --pulse-size 2

# PDF files would be generated in the tmp directory
```

- As another example, to visualize a 400 pps ACE signal (i.e. n-of-m algorithm) with n = 8 channels and m = 20 channels, use the following command

```Shell
python main.py \
    --mode visualize \
    --wav-list wav.list \
    --outdir tmp \
    --n-band 8 \
    --m-band 20 \
    --pps 400 \
    --pulse-type binary \
    --pulse-size 2
```

- Examples of output images with the speech signal /sa/ are as follows:
  - Left: CIS (400 pps, n = 20 channels, m = 20 channels)
  - Right: ACE (400 pps, n = 8 channels, m = 20 channels)

<img src="https://github.com/ashi-ta/ci_signal_simulator/blob/main/images/sa_cis.png" width="320px">　<img src="https://github.com/ashi-ta/ci_signal_simulator/blob/main/images/sa_ace.png" width="320px">

- In the case of the configuration depicted in the above figure, the frequency bands of electrode numbers 1 to 20 were set as follows:

| Electrode No. | Frequency band [Hz] |
|     :---:     |        :---:        |
|       1       |    6666.1-8000.0    |
|       2       |    5554.6-6666.1    |
|       3       |    4628.4-5554.6    |
|       4       |    3856.7-4628.4    |
|       5       |    3213.6-3856.7    |
|       6       |    2677.8-3213.6    |
|       7       |    2231.3-2677.8    |
|       8       |    1859.3-2231.3    |
|       9       |    1549.3-1859.3    |
|       10      |    1290.9-1549.3    |
|       11      |    1075.7-1290.9    |
|       12      |     896.3-1075.7    |
|       13      |     746.9-896.3     |
|       14      |     622.3-746.9     |
|       15      |     518.6-622.3     |
|       16      |     432.1-518.6     |
|       17      |     360.1-432.1     |
|       18      |     300.0-360.1     |
|       19      |     250.0-300.0     |
|       20      |     0.0-250.0       |

- Lower numbers on the electrodes indicate the basal region (higher frequency band) and higher numbers indicate the apical region (lower frequency band) of a cochlear

### Generate signals for F0 model training

- The following steps are not necessary if only analyzing and visualizing the pulse signal
- When preparing data for F0 model training, in addition to the above list files, a CSV file with the time and F0 labels corresponding to the WAV file is required
- For example, itemize the CSV file corresponding to `test1.wav` as follows and save it under the name `test1.csv`:

```Text
...
16.512290,144.28
16.515193,144.62
16.518095,147.54
16.520998,155.23
...
```

- where the first column is the time and the second column is the corresponding F0 label
- Next, create a list file itemizing the CSV files created above, as shown below:

```Text
/csv/path/test1.csv
/csv/path/test2.csv
/csv/path/test3.csv
...
```

- We trained the model with a [Kaldi](https://github.com/kaldi-asr/kaldi) format file, and hence, the output files are scp-, ark-, and text-format files.
- As an example, to prepare a 20-channel CIS signal at 400 pps for model training, use the following command

```Shell
# csv.list is the CSV list file prepared in the above preparation procedure

python main.py \
    --mode data_prep_for_f0 \
    --wav-list wav.list \
    --csv-list csv.list \
    --outdir tmp \
    --n-band 20 \
    --m-band 20 \
    --pps 400 \
    --pulse-type binary \
    --pulse-size 2
```

- `file.scp`, `file.ark`, and `text` files would be generated in the `tmp` directory and can be read via `kaldiio`

## Citation (TBD)

- TBD

```
Takanori Ashihara, Shigeto Furukawa, Makio Kashino, "F0 Estimation from Simulated Cochlear-implant Signals by Using a Deep Neural Networks", 2022
```
