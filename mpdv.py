# This script processes PDV (Photonic Doppler Velocimetry) data to generate spectrograms,
# apply filtering, and visualize raw, filtered, and cropped spectrograms.
# Input: PDV data file (e.g., .txt or .csv).
# Output: Spectrogram visualizations with labeled axes.

# Example usage:
# python mpdv.py --file LPC_071323_1_PDV.csv --win_size 4096

import numpy as np
import os, pathlib
import matplotlib.pyplot as plt
import csv
import scipy.signal as signal
import argparse

SPEED_OF_LIGHT = 2.998e8  # m/s
LASER_WAVELENGTH = 193391  # GHz

os.chdir(pathlib.Path(__file__).parent.resolve())
parser = argparse.ArgumentParser(description='Process PDV data files.')
parser.add_argument('--file', type=str, help='Path to the input data file.')
parser.add_argument('--win_size', type=int, default=2048, help='Window size for FFT. Default is 2048.')
args = parser.parse_args()

pfile = args.file
winSize = args.win_size

# This class extracts and processes raw signal data from a file.
class signal_ext():
    def __init__(self):
        try:
            print("Reading input file...")
            data = self.readFile(pfile)
        except FileNotFoundError:
            print(f"Error: File '{pfile}' not found.")
            exit(1)
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            exit(1)
        time, inten = self.cols(data)
        self.timeAxis = time
        self.rawSignal = np.array([time, inten])

        if len(self.rawSignal[1]) == 0:
            print("Error: Input signal is empty.")
            exit(1)

        self.N = len(time)
        DURATION = time[-1] - time[0]
        self.RATE = self.N/DURATION

    def readFile(self, path):
        data = []
        with open(path, 'r') as file:
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                data.append(row)
            if path.endswith(".txt"):           # Remove header lines for .txt files
                data = np.array(data[5:])
        return data
    
    def cols(self, data):
        time = np.zeros(len(data), dtype=float)
        inten = np.zeros(len(data), dtype=float)
        for i in range(len(data)):
            time[i] = data[i][0]
            inten[i] = data[i][1]
        return time, inten

# This class manages the figure layout for visualizing spectrograms and plots.    
class fig_mgr():
    def __init__(self):
        fig = plt.figure(figsize=(8, 8))
        self.subfigs = fig.subfigures(3, 1)

    def figure_setup(self, ncols):
        self.rawAx = self.subfigs[0].subplots(1, 1)
        self.croppedAx = self.subfigs[1].subplots(1, ncols)
        self.velAx = self.subfigs[2].subplots(1, ncols)

    def set_titles(self, axes, titles):
        """
        Set titles for the top figure's subplots.
        :param titles: List of titles for the subplots.
        """
        for ax, title in zip(axes, titles):
            ax.set_title(title)

    def set_axis_labels(self, axes, x_label, y_label):
        """
        Set x and y labels for the top figure's subplots.
        :param x_label: Label for the x-axis.
        :param y_label: Label for the y-axis.
        """
        for ax in axes:
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)

    def set_ticks(self, axes, x_ticks, x_labels, y_ticks, y_labels):
        """
        Set ticks and labels for the given axes.
        :param axes: List of Axes objects to update.
        :param x_ticks: Positions for x-axis ticks.
        :param x_labels: Labels for x-axis ticks.
        :param y_ticks: Positions for y-axis ticks.
        :param y_labels: Labels for y-axis ticks.
        """
        for ax in axes:
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels)

    def plot_data(self, axes, data, **kwargs):
        """
        Plot data on the given axes.
        :param axes: List of Axes objects to plot on.
        :param data: List of data arrays to plot.
        :param kwargs: Additional keyword arguments for the plot function.
        """
        for ax, d in zip(axes, data):
            ax.plot(d, **kwargs)

    def display_spectrograms(self, axes, spectrograms, cmap='viridis', **kwargs):
        """
        Display spectrograms on the given axes.
        :param axes: List of Axes objects to display on.
        :param spectrograms: List of spectrogram arrays to display.
        :param cmap: Colormap for the spectrograms.
        :param kwargs: Additional keyword arguments for the imshow function.
        """
        for ax, spec in zip(axes, spectrograms):
            ax.imshow(spec, cmap=cmap, interpolation='none', origin='lower', **kwargs)

    def highlight_roi(self, ax, xmin, xmax, ymin, ymax, shape, color="lime", label="Auto-ROI"):
        """
        Highlight a region of interest (ROI) on the given axis.
        :param ax: Axis object to highlight on.
        :param xmin: Minimum x value of the ROI.
        :param xmax: Maximum x value of the ROI.
        :param ymin: Minimum y value of the ROI.
        :param ymax: Maximum y value of the ROI.
        :param shape: Shape of the spectrogram (for normalization).
        :param color: Color of the ROI highlight.
        :param label: Label for the ROI.
        """
        ax.axvspan(
            xmin=xmin, xmax=xmax,
            ymin=ymin / shape[0], ymax=ymax / shape[0],
            color=color, fill=False, linestyle="-", label=label
        )
        ax.legend(loc="upper right")

# This class handles spectrogram generation, filtering, and cropping operations.
class specgram_transform():
    def __init__(self):
        # These initial parameters are used to change the frequency and time resolution of the generated spectrograms.
        self.win = signal.windows.hann(winSize)    
        hop = winSize//2                                # Default is 0.5 of winSize
        mfft = int(winSize * 1.00)                      # Use for zero padding, should be ~ 1.00 - 1.33 that of window size. Values >1.00 may make autocropping more difficult!

        self.SFT = signal.ShortTimeFFT(win=self.win, hop=hop, fs=sE.RATE, fft_mode='onesided', scale_to='magnitude', mfft=mfft)

        self.dictSx = {}                        # I keep track of the FFT outputs in these dicts
        self.dictSx_dB = {}

    def spectrogram(self, signal, index, power):
        # Signal refers to the time-space spectrum for which the FFT is to be run on.
        # Index refers to the dictionary index to output the linear and log spectrograms to.
        # Power refers to the magnitude of the spectrogram power
        self.dictSx.update({index: sT.SFT.spectrogram(signal)})
        self.dictSx_dB.update({index: 10*np.log10(np.fmax(self.dictSx[index], 10**-(power)))})

    def autoROI(self, Sx, Sx_dB):
        # Automatically detects regions of interest in the spectrogram based on intensity peaks.
        self.timeIntSpikes = np.mean(10**(Sx_dB/10), axis=0)                 # Determine max intensities for each time bin, normalize. Provides info on event start and end timewise
        self.timeIntSpikes = self.timeIntSpikes / np.max(self.timeIntSpikes)

        self.blFreqIntSpikes = np.mean((Sx[:,0:(Sx.shape[1]//10)]), axis=1)  # Determine max intensities for each freq bin, normalize. Provides info on frequency ranges for BASELINE frequencies
        self.blFreqIntSpikes = self.blFreqIntSpikes / np.max(self.blFreqIntSpikes)     # Only takes into account frequencies for first 10% of time points

        self.avgFreq = np.mean((Sx), axis=1)                                 # Now determine max intensities for reach freq bin, normalize, but take into account all of the data.
        self.avgFreq = self.avgFreq / np.max(self.avgFreq)
        self.avgSignalFreq = np.clip(self.avgFreq - self.blFreqIntSpikes, a_min=0, a_max=None, out=None)                       # This gives a better idea of where the ROI is in relation to the baselines; Baselines should be negative, the target signal should be positive
        self.avgSignalFreq = self.avgSignalFreq / np.max(self.avgSignalFreq)

        self.horz_crop_peaks = signal.find_peaks(self.avgSignalFreq, height=0.01)
        self.time_peaks = signal.find_peaks(self.timeIntSpikes, height = 0.2, distance=sT.dictSx_dB[2].shape[1]//20)
        self.freq_peaks = signal.find_peaks(self.blFreqIntSpikes, height=0.1, distance=sT.dictSx_dB[2].shape[0]//20) # Baseline frequency peaks not counted if within 5% index of another peak

        # Filter out frequency peaks that are within the first 5% of the frequency range
        self.freq_peaks = tuple(
            peak for peak in self.freq_peaks[0] if peak > (Sx.shape[0] * 0.05)
        )

    def notch(self, sig):
        self.filtered_signal = sig
        for n in range(len(sT.freq_peaks)):
            b, a = signal.iirnotch(freqAxis[sT.freq_peaks[n]]*1e9, Q=75, fs=sE.RATE)       # If notch is too broad, raise the Q value. Default is ~30, >=100 is likely too high
            self.filtered_signal = signal.filtfilt(b, a, self.filtered_signal)
        self.filtered_signal[np.isnan(self.filtered_signal)] = 0
        

    def bandpass(self, Sx, sig):
        """
        Apply a bandpass filter to the input signal.
        The filter is designed based on detected frequency peaks and cropping parameters.
        """
        self.filtered_signal = sig

        # Calculate the lower and upper frequency bounds for the bandpass filter
        lower_bound = freqAxis[
            np.clip(
                sT.freq_peaks[0] - (Sx.shape[0] // 100),
                a_min=0,
                a_max=None
            )
        ] * 1e9  # Convert to Hz

        upper_bound = freqAxis[
            np.clip(
                sT.horz_crop_peaks[0][-1] + (Sx.shape[0] // 100),
                a_min=None,
                a_max=Sx.shape[0]
            )
        ] * 1e9  # Convert to Hz

        # Design a 4th-order Butterworth bandpass filter
        b, a = signal.butter(4, [lower_bound, upper_bound], fs=sE.RATE, btype='bandpass')

        # Apply the filter to the signal
        self.filtered_signal = signal.filtfilt(b, a, self.filtered_signal)

        # Replace NaN values in the filtered signal with 0
        self.filtered_signal[np.isnan(self.filtered_signal)] = 0

    def crop(self, Sx_dB):
        #print(np.clip((sT.time_peaks[0][0]-(Sx_dB.shape[1]//10)), a_min=0, a_max=None, out=None))
        #print(np.clip((sT.time_peaks[0][-1]+(Sx_dB.shape[1]//10)), a_min=None, a_max=Sx_dB.shape[1], out=None))
        self.xmin = np.clip((sT.time_peaks[0][0]-(Sx_dB.shape[1]//10)), a_min=0, a_max=None, out=None)+1
        self.xmax = np.clip((sT.time_peaks[0][-1]+(Sx_dB.shape[1]//5)), a_min=None, a_max=Sx_dB.shape[1], out=None)-1
        self.ymin = np.clip((sT.freq_peaks[0]-(Sx_dB.shape[0]//100)), a_min=0, a_max=None, out=None)+1
        self.ymax = np.clip((sT.horz_crop_peaks[0][-1]+(Sx_dB.shape[0]//50)), a_min=None, a_max=Sx_dB.shape[0], out=None)-1
    
    def velCalc(self, Sx, ymin, ymax, baselineIndex):
        max_freqs = []
        # Iterate over each time bin in the cropped spectrogram
        globalMax = np.max(Sx)

        
        for time_bin in Sx.T:
            # Crop the frequency axis to match the spectrogram's cropped region
            cropped_freqAxis = freqAxis[ymin:ymax]
            if(np.max(time_bin) >= 0.005*globalMax):           # Only consider time bins with significant intensity relative to the global maximum
                max_freqs.append(cropped_freqAxis[np.argmax(time_bin)])
            else:
                max_freqs.append(np.nan)
                   
        # Adjust the frequency values relative to the baseline frequency
        beatFreq = np.array(max_freqs) - freqAxis[sT.freq_peaks[baselineIndex]]

        velocity = beatFreq * SPEED_OF_LIGHT / (2 * LASER_WAVELENGTH)
        #velocity = np.where(velocity == 0, np.nan, velocity)            # Replace zero values with NaN to avoid plotting invalid data
        #velocity = np.clip(velocity, a_min=-50, a_max=None, out=None)     # Clip negative values to zero
        return velocity
       
    def demux(self, Sx, muxed_signal):
        """
        Separate the signal into multiple frequency components based on detected frequency peaks. Run bandpass and notch on each component then create new dict for cropped, demuxed signals/FFTs.
        """
        mxsig = muxed_signal

        self.demuxed_signals = {}
        self.demuxed_fft = {}
        self.demuxed_fft_dB = {}

        self.mx_xmin = sT.xmin
        self.mx_xmax = sT.xmax

        self.mx_ymins = []
        self.mx_ymaxs = []
        for n in range(len(sT.freq_peaks)):
            # Calculate the lower and upper frequency bounds for the bandpass filter
            lower_bound = freqAxis[
                np.clip(
                    sT.freq_peaks[n] - (Sx.shape[0] // 100),
                    a_min=0,
                    a_max=None
                )
            ] * 1e9
            if(n == len(sT.freq_peaks)-1):               # If this is the last frequency peak, use the last detected peak for the upper bound
                upper_bound = freqAxis[
                    np.clip(
                        sT.horz_crop_peaks[0][-1] + (Sx.shape[0] // 20),
                        a_min=None,
                        a_max=Sx.shape[0]
                    )
                ] * 1e9
                self.mx_ymin = np.clip((sT.freq_peaks[n]- (Sx.shape[0] // 100)), a_min=0, a_max=None, out=None)+1
                self.mx_ymax = np.clip(sT.horz_crop_peaks[0][-1] + (Sx.shape[0] // 20), a_min=None, a_max=Sx.shape[0], out=None)-1
            else:                                        # Otherwise, use the next detected peak for the upper bound
                upper_bound = freqAxis[
                    np.clip(
                        sT.freq_peaks[n+1] - (Sx.shape[0] // 20),
                        a_min=None,
                        a_max=Sx.shape[0]
                    )
                ] * 1e9
                self.mx_ymin = np.clip((sT.freq_peaks[n]- (Sx.shape[0] // 100)), a_min=0, a_max=None, out=None)+1
                self.mx_ymax = np.clip((sT.freq_peaks[n+1]- (Sx.shape[0] // 20)), a_min=None, a_max=Sx.shape[0], out=None)-1

            self.mx_ymins.append(self.mx_ymin)
            self.mx_ymaxs.append(self.mx_ymax)    
            # Design a 4th-order Butterworth bandpass filter
            b, a = signal.butter(4, [lower_bound, upper_bound], fs=sE.RATE, btype='bandpass')
            dmx_temp = signal.filtfilt(b, a, mxsig)
            b, a = signal.iirnotch(freqAxis[sT.freq_peaks[n]]*1e9, Q=200, fs=sE.RATE)       # If notch is too broad, raise the Q value. Default is ~30, >=100 is likely too high
            dmx_temp = signal.filtfilt(b, a, dmx_temp)

            # Replace NaN values in the filtered signal with 0
            dmx_temp[np.isnan(dmx_temp)] = 0
        
            # Store the filtered signal in the demuxed_signals dictionary
            self.demuxed_signals[f"Demuxed_Signal_{n+1}"] = dmx_temp

            test = sT.SFT.spectrogram(dmx_temp)
            test = test[np.clip(sT.mx_ymin, 0, None):sT.mx_ymax, sT.mx_xmin:sT.mx_xmax]
            
            self.demuxed_fft.update({n+1: test})
            self.demuxed_fft_dB.update({n+1: 10*np.log10(np.fmax(self.demuxed_fft[n+1], 10**-(9)))})

            colors = ['pink', 'red', 'magenta', 'yellow', 'orange']  # Define a list of colors
            color = colors[n % len(colors)]  # Cycle through colors based on n
            fM.highlight_roi(
                fM.rawAx,
                xmin=sT.mx_xmin, xmax=sT.mx_xmax,
                ymin=sT.mx_ymin, ymax=sT.mx_ymax,
                shape=sT.dictSx_dB[2].shape,
                color=color, label=f'Probe # {n+1}'
            )




if __name__ == '__main__':
    sE = signal_ext()
    sT = specgram_transform()
    fM = fig_mgr()

    sT.spectrogram(sE.rawSignal[1], 2, 9)       # FFT on intensities, and output magnitudes of frequencies over time to position 2 in the dictionary w/ a power of 9
    sT.autoROI(sT.dictSx[2], sT.dictSx_dB[2])
    numFreqBins = sT.dictSx_dB[2].shape[0] # num rows
    frequencies = np.abs(np.fft.fftfreq(len(sT.win), d=1/sE.RATE))
    freqAxis = np.flip((frequencies / 1e9)[numFreqBins//1:])

    print(f"Number of Detected Baselines: {len(sT.freq_peaks)}")

    if len(sT.freq_peaks) > 1:               # If more than one baseline frequency is detected, demultiplex the signal
        fM.figure_setup(len(sT.freq_peaks))  # Create subfigures for each detected frequency peak
        print("More than one baseline frequency detected. Demultiplexing...")
        sT.crop(sT.dictSx_dB[2])
        sT.demux(sT.dictSx_dB[2], sE.rawSignal[1])
        fM.rawAx.imshow(sT.dictSx_dB[2], cmap='viridis', interpolation='none', origin='lower')
        fM.display_spectrograms(
                fM.croppedAx,
                [sT.demuxed_fft_dB[n] for n in range(1, len(sT.freq_peaks)+1)],
            )
        

        time_zero_index = np.argmin(np.abs(sE.timeAxis))  # Find the index closest to timeAxis = 0
        time_zero_position = time_zero_index / len(sE.timeAxis) * sT.dictSx_dB[2].shape[1]  # Normalize to spectrogram axis
        fM.rawAx.axvline(x=time_zero_position, color='lime', linestyle='--', label='Time = 0')
        
        for n in range(len(sT.freq_peaks)):
            vel = sT.velCalc(sT.demuxed_fft[n+1], sT.mx_ymins[n], sT.mx_ymaxs[n], n)
            #fM.velAx[n].scatter(np.linspace(sT.mx_xmin, sT.mx_xmax, len(vel)), vel, s=1, label="Velocity (m/s)", color="red")
            fM.velAx[n].plot(vel, label="Velocity (m/s)", color="red")

        fM.set_titles(fM.velAx, [f"Velocity for Probe #{n+1}" for n in range(len(sT.freq_peaks))])
        fM.set_axis_labels(fM.velAx, "Time (µs)", "Velocity (m/s)")


        num_time_ticks = 10
        num_freq_ticks = 15

        xp = np.linspace(0, sT.dictSx_dB[2].shape[1], sT.dictSx_dB[2].shape[1])
        yp = np.linspace(0, sT.dictSx_dB[2].shape[0], sT.dictSx_dB[2].shape[0])
        tp = np.linspace(sE.timeAxis[0], sE.timeAxis[-1], xp.shape[0])
        fp = np.linspace(freqAxis[0], freqAxis[-1], yp.shape[0])
        timeAxesPos = np.linspace(0, sT.dictSx_dB[2].shape[1], num_time_ticks)
        timeAxesLabels = [f"{val:.0f}" for val in np.interp(timeAxesPos, xp, tp) * 1e6]
        freqAxesPos = np.linspace(0, sT.dictSx_dB[2].shape[0], num_freq_ticks)
        freqAxesLabels = [f"{val:.1f}" for val in np.interp(freqAxesPos, yp, fp)]
        
        # Generate cropped time axis positions and labels
        timeAxesPos_cropped = np.linspace(0, sT.xmax - sT.xmin, num_time_ticks)
        timeAxesLabels_cropped = [f"{val:.1f}" for val in np.interp(timeAxesPos_cropped + sT.xmin, xp, tp) * 1e6]

        # Generate cropped frequency axis positions and labels
        for n in range(len(sT.freq_peaks)):
            freqAxesPos_cropped = np.linspace(0, sT.mx_ymaxs[n] - sT.mx_ymins[n], num_freq_ticks)
            freqAxesLabels_cropped = [f"{val:.1f}" for val in np.interp(freqAxesPos_cropped + sT.mx_ymins[n], yp, fp)]
            fM.set_ticks(
                [fM.croppedAx[n]],  # Apply to the cropped spectrograms
                timeAxesPos_cropped, timeAxesLabels_cropped,
                freqAxesPos_cropped, freqAxesLabels_cropped
            )


        fM.set_ticks(
            [fM.rawAx],  # Wrap in a list to ensure compatibility
            timeAxesPos, timeAxesLabels,
            freqAxesPos, freqAxesLabels
        )

        for ax in fM.velAx:
            ax.set_xticks(timeAxesPos_cropped + sT.xmin)
            ax.set_xticklabels(timeAxesLabels_cropped)





    else:                      # If only one baseline frequency is detected, proceed with single baseline frequency
        fM.figure_setup(1)
        print("Proceeding with single baseline frequency.")
        fM.rawAx.imshow(sT.dictSx_dB[2], cmap='viridis', interpolation='none', origin='lower')
        sT.bandpass(sT.dictSx_dB[2], sE.rawSignal[1])
        sT.notch(sT.filtered_signal)
        sT.spectrogram(sT.filtered_signal, 3, 9)
        sT.autoROI(sT.dictSx[3], sT.dictSx_dB[3])
        sT.crop(sT.dictSx_dB[3])
        Sx3_dB_cropped = sT.dictSx_dB[3][np.clip(sT.ymin, 0, None):sT.ymax, sT.xmin:sT.xmax]
        Sx3_cropped = sT.dictSx[3][np.clip(sT.ymin, 0, None):sT.ymax, sT.xmin:sT.xmax]

        fM.croppedAx.imshow(Sx3_dB_cropped, cmap='viridis', interpolation='none', origin='lower')
        
        # Add a vertical red dashed line at timeAxis = 0 to topFig[1] and topFig[2]
        time_zero_index = np.argmin(np.abs(sE.timeAxis))  # Find the index closest to timeAxis = 0
        time_zero_position = time_zero_index / len(sE.timeAxis) * sT.dictSx_dB[2].shape[1]  # Normalize to spectrogram axis
        fM.rawAx.axvline(x=time_zero_position, color='lime', linestyle='--', label='t=0')
        fM.rawAx.legend(loc="upper left")

        vel = sT.velCalc(Sx3_cropped, sT.ymin, sT.ymax, 0)
        fM.velAx.scatter(np.linspace(sT.xmin, sT.xmax, len(vel)), vel, s=1, label="Velocity (m/s)", color="red")

        num_time_ticks = 10
        num_freq_ticks = 15

        xp = np.linspace(0, sT.dictSx_dB[2].shape[1], sT.dictSx_dB[2].shape[1])
        yp = np.linspace(0, sT.dictSx_dB[2].shape[0], sT.dictSx_dB[2].shape[0])
        tp = np.linspace(sE.timeAxis[0], sE.timeAxis[-1], xp.shape[0])
        fp = np.linspace(freqAxis[0], freqAxis[-1], yp.shape[0])
        timeAxesPos = np.linspace(0, sT.dictSx_dB[2].shape[1], num_time_ticks)
        timeAxesLabels = [f"{val:.0f}" for val in np.interp(timeAxesPos, xp, tp) * 1e6]
        freqAxesPos = np.linspace(0, sT.dictSx_dB[2].shape[0], num_freq_ticks)
        freqAxesLabels = [f"{val:.1f}" for val in np.interp(freqAxesPos, yp, fp)]
        
        # Generate cropped time axis positions and labels
        timeAxesPos_cropped = np.linspace(0, sT.xmax - sT.xmin, num_time_ticks)
        timeAxesLabels_cropped = [f"{val:.2f}" for val in np.interp(timeAxesPos_cropped + sT.xmin, xp, tp) * 1e6]

        # Generate cropped frequency axis positions and labels
        freqAxesPos_cropped = np.linspace(0, sT.ymax - sT.ymin, num_freq_ticks)
        freqAxesLabels_cropped = [f"{val:.1f}" for val in np.interp(freqAxesPos_cropped + sT.ymin, yp, fp)]

        #fM.set_titles(fM.rawAx, ["Raw Signal Spectrogram", "Filtered Signal Spectrogram", "Filtered Signal Spectrogram (Cropped)"])
        #fM.set_axis_labels(fM.rawAx, "Time (µs)", "Frequency (GHz)")
        fM.velAx.set_title("Velocity over Time")
        fM.velAx.set_xlabel("Time (µs)")
        fM.velAx.set_ylabel("Velocity (m/s)")
        
        
        fM.set_ticks(
            [fM.rawAx],  # Apply to the first two subplots
            timeAxesPos, timeAxesLabels,
            freqAxesPos, freqAxesLabels
        )
        
        fM.velAx.set_xticks(timeAxesPos_cropped+sT.xmin)
        fM.velAx.set_xticklabels(timeAxesLabels_cropped)
        
        fM.highlight_roi(
            fM.rawAx,
            xmin=sT.xmin, xmax=sT.xmax,
            ymin=sT.ymin, ymax=sT.ymax,
            shape=sT.dictSx_dB[3].shape,
            color="orange", label="Auto-ROI"
        )        
    plt.show()
