from obspy import UTCDateTime
from obspy.clients.filesystem import sds
import numpy as np
import logging
import os
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
from scipy.stats import kurtosis, skew
from scipy.signal import hilbert
import warnings
from obspy.clients.fdsn import Client
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define parameters
root_dir = "./Grenzgletscher_fk/"
input_file = "./Grenzgletscher_fk/2025_fk_trigger.csv"
output_file = "./Grenzgletscher_fk/2025_features.csv"
window_before = 1    # seconds before trigger
window_after = 5     # seconds after trigger

# Velocity model parameters
p_velocity = 3.8  # km/s for P-waves
s_velocity = 1.8  # km/s for S-waves

cl = sds.Client(sds_root=root_dir)
clw = Client("http://tarzan.geophysik.uni-muenchen.de")

def calculate_incidence_angles_from_slowness(slowness_data):
    """Calculate incidence angles from slowness values using velocity model"""
    if slowness_data is None or len(slowness_data) == 0:
        return None, None
    
    incidence_angles = []
    wave_types = []
    
    for s in slowness_data:
        if np.isnan(s):
            incidence_angles.append(np.nan)
            wave_types.append("Unknown")
            continue
            
        if s < 0.3:  # P-wave region
            sin_i = min(p_velocity * s, 0.99)
            angle = np.degrees(np.arcsin(sin_i))
            wave_type = "P"
        elif s <= 0.6:  # S-wave region
            if s_velocity * s > 0.99:
                # Scale between 60-85 degrees based on the slowness value
                angle = 60 + 25 * (s - 0.2) / 0.3
            else:
                sin_i = min(s_velocity * s, 0.99)
                angle = np.degrees(np.arcsin(sin_i))
            wave_type = "S"
        else:
            # For values outside velocity model assumptions
            angle = np.nan
            wave_type = "Unknown"
            
        incidence_angles.append(angle)
        wave_types.append(wave_type)
    
    return np.array(incidence_angles), np.array(wave_types)

def map_array_to_waveform_indices(array_data, array_sr, waveform_sr, waveform_length):
    # Map array processing data to waveform sample indices using interpolation
    if array_data is None or len(array_data) == 0:
        return None
    
    try:
        array_times = np.arange(len(array_data)) / array_sr
        waveform_times = np.arange(waveform_length) / waveform_sr
        
        if len(array_data) == 1:
            return np.full(waveform_length, array_data[0])
        
        interp_func = interp1d(array_times, array_data, kind='linear', 
                              fill_value='extrapolate', bounds_error=False)
        mapped_data = interp_func(waveform_times)
        
        # Handle NaN values
        if np.any(np.isnan(mapped_data)):
            mask = ~np.isnan(mapped_data)
            if np.any(mask):
                first_valid = np.where(mask)[0][0]
                mapped_data[:first_valid] = mapped_data[first_valid]
                for i in range(1, len(mapped_data)):
                    if np.isnan(mapped_data[i]):
                        mapped_data[i] = mapped_data[i-1]
        
        return mapped_data
        
    except Exception as e:
        logger.warning(f"Error in array data mapping: {e}")
        return np.full(waveform_length, np.mean(array_data))

def analyze_slowness_during_signal(ar_slowness, ar_back_azimuth, ar_incidence_angle, 
                                 envelope, trigger_sample, sample_rate):

   # Analyze slowness changes from trigger until amplitude drops below 2% of max.
    features = {}
    
    if ar_slowness is None or len(ar_slowness) == 0:
        # No slowness data available
        for key in ['slowness_min_during_signal', 'slowness_mean_during_signal', 
                   'slowness_std_during_signal', 'slowness_change_during_signal',
                   'back_azimuth_at_min_slowness', 'incidence_angle_at_min_slowness']:
            features[key] = np.nan
        return features
    
    # Find signal activity window
    envelope_max = np.max(envelope)
    threshold = envelope_max * 0.02  # 2% threshold
    
    # Find start and end of signal activity
    above_threshold = envelope > threshold
    if not np.any(above_threshold):
        # No significant signal
        for key in ['slowness_min_during_signal', 'slowness_mean_during_signal', 
                   'slowness_std_during_signal', 'slowness_change_during_signal',
                   'back_azimuth_at_min_slowness', 'incidence_angle_at_min_slowness']:
            features[key] = np.nan
        return features
    
    signal_indices = np.where(above_threshold)[0]
    signal_start = max(0, signal_indices[0])
    signal_end = min(len(ar_slowness), signal_indices[-1] + 1)
    
    # Extract slowness during signal activity
    slowness_during_signal = ar_slowness[signal_start:signal_end]
    
    # Remove NaN values
    valid_mask = ~np.isnan(slowness_during_signal)
    if not np.any(valid_mask):
        for key in ['slowness_min_during_signal', 'slowness_mean_during_signal', 
                   'slowness_std_during_signal', 'slowness_change_during_signal',
                   'back_azimuth_at_min_slowness', 'incidence_angle_at_min_slowness']:
            features[key] = np.nan
        return features
    
    valid_slowness = slowness_during_signal[valid_mask]
    valid_indices = np.where(valid_mask)[0] + signal_start
    
    # Slowness features during signal
    features['slowness_min_during_signal'] = np.min(valid_slowness)
    features['slowness_mean_during_signal'] = np.mean(valid_slowness)
    features['slowness_std_during_signal'] = np.std(valid_slowness)
    features['slowness_change_during_signal'] = np.max(valid_slowness) - np.min(valid_slowness)
    
    # Find index of minimum slowness
    min_slowness_idx_in_valid = np.argmin(valid_slowness)
    min_slowness_global_idx = valid_indices[min_slowness_idx_in_valid]
    
    # Back azimuth at minimum slowness
    if (ar_back_azimuth is not None and len(ar_back_azimuth) > min_slowness_global_idx):
        features['back_azimuth_at_min_slowness'] = ar_back_azimuth[min_slowness_global_idx]
    else:
        features['back_azimuth_at_min_slowness'] = np.nan
    
    # Incidence angle at minimum slowness
    if (ar_incidence_angle is not None and len(ar_incidence_angle) > min_slowness_global_idx):
        features['incidence_angle_at_min_slowness'] = ar_incidence_angle[min_slowness_global_idx]
    else:
        features['incidence_angle_at_min_slowness'] = np.nan
    
    return features

def extract_features(trace, ar_back_azimuth, ar_incidence_angle, ar_slowness, trigger_time):
    # Extract all required features from the data
    
    if trace is None or not hasattr(trace, 'data') or len(trace.data) == 0:
        return {}
    
    data = trace.data.astype(np.float64)
    data = data - np.mean(data)  # Remove DC
    sample_rate = trace.stats.sampling_rate
    features = {}
    
    # Basic waveform features
    features['waveform_rms'] = np.sqrt(np.mean(np.square(data)))
    features['waveform_peak_to_peak'] = np.max(data) - np.min(data)
    features['waveform_abs_energy'] = np.sum(np.abs(data))
    features['waveform_max_abs'] = np.max(np.abs(data))
    features['waveform_mean_abs'] = np.mean(np.abs(data))
    features['waveform_skewness'] = skew(data)
    features['waveform_kurtosis'] = kurtosis(data)
    features['waveform_std'] = np.std(data)
    
    # Zero crossing rate
    zero_crossings = np.where(np.diff(np.signbit(data)))[0]
    features['zero_crossing_rate'] = len(zero_crossings) / (len(data) / sample_rate)
    
    # Spectral features
    if len(data) >= 4:
        windowed_data = data * signal.windows.hann(len(data))
        fft = np.abs(np.fft.rfft(windowed_data))
        freqs = np.fft.rfftfreq(len(data), d=1.0/sample_rate)
        
        if len(fft) > 1:
            fft = fft[1:]  # Remove DC
            freqs = freqs[1:]
        
        if len(fft) > 0 and np.sum(fft) > 1e-10:
            total_power = np.sum(fft)
            features['spectral_centroid'] = np.sum(freqs * fft) / total_power
            features['dominant_freq'] = freqs[np.argmax(fft)]
            features['spectral_spread'] = np.sqrt(np.sum(((freqs - features['spectral_centroid']) ** 2) * fft) / total_power)
            
            # Spectral rolloff
            cumulative_power = np.cumsum(fft)
            rolloff_idx = np.where(cumulative_power >= 0.85 * total_power)[0]
            features['spectral_rolloff'] = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
            
            # Spectral flatness
            geometric_mean = np.exp(np.mean(np.log(fft + 1e-10)))
            features['spectral_flatness'] = geometric_mean / np.mean(fft)
            
            # Frequency band ratios
            fft_power = fft**2
            total_power_squared = np.sum(fft_power)
            for low, high in [(1, 5), (5, 10), (10, 15), (15, 20)]:
                band_indices = np.logical_and(freqs >= low, freqs <= high)
                if np.any(band_indices):
                    band_energy = np.sum(fft_power[band_indices])
                    features[f'spec_band_{low}_{high}_ratio'] = band_energy / total_power_squared
                else:
                    features[f'spec_band_{low}_{high}_ratio'] = 0.0
        else:
            # Default spectral features
            for key in ['spectral_centroid', 'dominant_freq', 'spectral_spread', 'spectral_rolloff', 'spectral_flatness']:
                features[key] = 0.0
            for low, high in [(1, 5), (5, 10), (10, 15), (15, 20)]:
                features[f'spec_band_{low}_{high}_ratio'] = 0.0
    
    # Envelope and duration features
    try:
        analytic_signal = hilbert(data)
        envelope = np.abs(analytic_signal)
        
        # Smooth envelope
        from scipy.ndimage import uniform_filter1d
        smooth_window = max(1, int(sample_rate * 0.05))
        smoothed_envelope = uniform_filter1d(envelope, size=smooth_window)
        
        envelope_max = np.max(smoothed_envelope)
        
        if envelope_max > 0:
            # Duration features with 5% and 2% thresholds
            for threshold, name in [(0.05, '5_percent'), (0.02, '2_percent')]:
                envelope_threshold = envelope_max * threshold
                above_threshold = smoothed_envelope > envelope_threshold
                
                if np.any(above_threshold):
                    duration_indices = np.where(above_threshold)[0]
                    duration_sec = (duration_indices[-1] - duration_indices[0]) / sample_rate
                    features[f'duration_{name}'] = duration_sec
                else:
                    features[f'duration_{name}'] = 0.0
            
            # Rise time
            above_start = smoothed_envelope > (envelope_max * 0.05)
            above_end = smoothed_envelope > (envelope_max * 0.8)
            
            if np.any(above_start) and np.any(above_end):
                rise_start_idx = np.where(above_start)[0][0]
                rise_end_idx = np.where(above_end)[0][0]
                features['envelope_rise_time'] = max(0, (rise_end_idx - rise_start_idx) / sample_rate)
            else:
                features['envelope_rise_time'] = 0.0
            
            # Peak position and envelope shape
            peak_idx = np.argmax(smoothed_envelope)
            features['peak_position_normalized'] = peak_idx / len(smoothed_envelope)
            features['envelope_skewness'] = skew(smoothed_envelope)
            features['envelope_kurtosis'] = kurtosis(smoothed_envelope)
        else:
            # Zero envelope case
            for name in ['5_percent', '2_percent']:
                features[f'duration_{name}'] = 0.0
            features['envelope_rise_time'] = 0.0
            features['peak_position_normalized'] = 0.5
            features['envelope_skewness'] = 0.0
            features['envelope_kurtosis'] = 0.0
        
        # Improved slowness analysis during signal activity
        trigger_sample = int(window_before * sample_rate)
        slowness_features = analyze_slowness_during_signal(
            ar_slowness, ar_back_azimuth, ar_incidence_angle,
            smoothed_envelope, trigger_sample, sample_rate
        )
        features.update(slowness_features)
        
    except Exception as e:
        logger.warning(f"Error computing envelope/slowness features: {e}")
        # Set defaults
        for name in ['5_percent', '2_percent']:
            features[f'duration_{name}'] = 0.0
        features['envelope_rise_time'] = 0.0
        features['peak_position_normalized'] = 0.5
        features['envelope_skewness'] = 0.0
        features['envelope_kurtosis'] = 0.0
        for key in ['slowness_min_during_signal', 'slowness_mean_during_signal', 
                   'slowness_std_during_signal', 'slowness_change_during_signal',
                   'back_azimuth_at_min_slowness', 'incidence_angle_at_min_slowness']:
            features[key] = np.nan
    
    return features

def main():
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load trigger times
    try:
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file)
            time_column = df.columns[0]
            trigger_times = [UTCDateTime(ts) for ts in df[time_column]]
        else:
            with open(input_file, 'r') as f:
                trigger_times = [UTCDateTime(line.strip()) for line in f if line.strip()]
        logger.info(f"Loaded {len(trigger_times)} trigger times")
    except Exception as e:
        logger.error(f"Error loading trigger file: {e}")
        return
    
    all_features = []
    
    # Process each event
    for i, event_time in enumerate(trigger_times):
        try:
            logger.info(f"Processing event {i+1}/{len(trigger_times)} at {event_time}")
            event_features = {"event_time": str(event_time)}

            # Get waveform data
            try:
                st = clw.get_waveforms(network="XG", station="UP1", location="*", channel="??Z", 
                                       starttime=(event_time-window_before), 
                                       endtime=event_time+window_after)
            except Exception as e:
                logger.error(f"Error getting waveform data: {e}")
                continue
            
            # Get array data
            try:
                ar = cl.get_waveforms(network="XG", station="UP1", location="*", channel="ZG?", 
                                     starttime=(event_time-window_before), 
                                     endtime=event_time+window_after)
            except Exception as e:
                logger.warning(f"Error getting array data: {e}")
                ar = []
            
            if len(st) == 0:
                continue
                
            # Preprocess seismic waveforms only
            st.detrend("linear")
            st.taper(type='cosine', max_percentage=0.05)
            st.filter("bandpass", freqmin=1, freqmax=20)
            
            vert_traces = st.select(component="Z")
            if len(vert_traces) == 0:
                continue
                
            vert_tr = vert_traces[0]
            
            # Process array data
            back_azimuth = incidence_angle = slowness = None
            if len(ar) > 0:
                for channel_code, var_name in [("ZGS", "back_azimuth"), ("ZGA", "slowness")]:
                    traces = [tr for tr in ar if tr.stats.channel == channel_code]
                    if traces:
                        raw_data = traces[0].data
                        raw_sr = traces[0].stats.sampling_rate
                        mapped_data = map_array_to_waveform_indices(
                            raw_data, raw_sr, vert_tr.stats.sampling_rate, len(vert_tr.data)
                        )
                        if var_name == "back_azimuth":
                            back_azimuth = mapped_data
                        elif var_name == "slowness":
                            slowness = mapped_data
                
                # Calculate incidence angles from slowness instead of reading from ZGI
                if slowness is not None:
                    incidence_angle, wave_types = calculate_incidence_angles_from_slowness(slowness)
            
            # Extract features
            extracted_features = extract_features(vert_tr, back_azimuth, incidence_angle, slowness, event_time)
            
            if extracted_features:
                event_features.update(extracted_features)
                all_features.append(event_features)
                
        except Exception as e:
            logger.error(f"Error processing event at {event_time}: {e}")
            continue
    
    # Save results
    if all_features:
        df = pd.DataFrame(all_features)
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(all_features)} feature sets to {output_file}")
        logger.info(f"Total features: {len(df.columns)-1}")
    else:
        logger.error("No features extracted")

if __name__ == "__main__":
    main()