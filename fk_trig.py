from obspy import *
from obspy.clients.filesystem import sds
from obspy.clients.fdsn import Client
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
import numpy as np
import logging
import os
from datetime import timedelta
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

# Set better default style for matplotlib with larger fonts
plt.style.use('seaborn-v0_8')
mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 12  # Base font size
mpl.rcParams['axes.titlesize'] = 16  # Subplot titles
mpl.rcParams['axes.labelsize'] = 14  # Axis labels
mpl.rcParams['xtick.labelsize'] = 12  # X-axis tick labels
mpl.rcParams['ytick.labelsize'] = 12  # Y-axis tick labels
mpl.rcParams['legend.fontsize'] = 12  # Legend font size
mpl.rcParams['figure.titlesize'] = 20  # Main figure title

# Add logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

out_file = "./Grenzgletscher_fk/fk_trigger.csv"
root_dir = "./Grenzgletscher_fk/"
save_path = "./Grenzgletscher_fk/trig_figs/"

# Create directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)

cl = sds.Client(sds_root=root_dir)

ostart = start = UTCDateTime(2024, 3, 19)
end = UTCDateTime(2024, 3, 23)
trig_times = []

# Debug info
logger.info(f"Starting trigger detection from {ostart} to {end}")
trigger_count = 0

# Thresholds for detection
relp_threshold = 0.6
slow_threshold = 0.5
min_trigger_separation = 2  # Minimum seconds between triggers

last_trigger_abs_time = None

while start + 1*3600 < end:
    endy = start + 1*3600
    try:
        logger.info(f"Processing window: {start} - {endy}")
        st = cl.get_waveforms(network="XG", station="UP1", location="", channel="ZG?", starttime=start, endtime=endy)
        st.merge()
        
        # Check for required channels
        channels = [tr.stats.channel for tr in st]
        if "ZGC" not in channels or "ZGA" not in channels:
            logger.warning(f"Missing required channels. Available: {channels}")
            start += 1*3600
            continue
        
        relp = st.select(channel="ZGC")[0]
        slow = st.select(channel="ZGA")[0]
        
        window_triggers = 0
        
        for i in range(1, relp.stats.npts):
            # Check if conditions are met
            if relp.data[i] > relp_threshold and slow.data[i] < slow_threshold:
                # Check for state transition
                if i > 5 and (
                    (np.mean(relp.data[i-1:i]) < relp_threshold) or 
                    (np.mean(slow.data[i-1:i]) > slow_threshold)
                ):
                    trig_time = relp.times(reftime=ostart)[i]
                    abs_time = ostart + trig_time
                    
                    # Check minimum separation
                    if last_trigger_abs_time is None or (abs_time - last_trigger_abs_time) > min_trigger_separation:
                        trig_times.append(trig_time)
                        logger.info(f"Trigger detected at index {i}: relative time={trig_time}, absolute time={abs_time}")
                        window_triggers += 1
                        last_trigger_abs_time = abs_time
                    else:
                        logger.info(f"Skipping close trigger at {abs_time} (too close to previous)")
        
        logger.info(f"Found {window_triggers} triggers in this window")
        trigger_count += window_triggers
        start += 1*3600
    except Exception as e:
        logger.error(f"Error processing window {start}-{endy}: {e}")
        start += 1*3600
        continue

logger.info(f"Total triggers detected: {trigger_count}")
logger.info(f"Writing triggers to {out_file}")

# Add validation before writing to CSV
valid_triggers = []
for j in trig_times:
    trigger_time = ostart + j
    valid_triggers.append(trigger_time)

# Write triggers to file
with open(out_file, "w") as fo:
    for trigger_time in valid_triggers:
        fo.write("%s\n" % trigger_time)

logger.info(f"Written {len(valid_triggers)} triggers to CSV file")

# Process individual events
all_baz = []
all_slow = []
all_rp = []
clw = Client("http://tarzan.geophysik.uni-muenchen.de")

# Define the global colormap
global_cmap = plt.cm.viridis

# Define wave velocities for incidence angle calculation
p_velocity = 3.8  # km/s for P-waves
s_velocity = 1.8  # km/s for S-waves

logger.info(f"Processing {len(trig_times)} individual events")

#Track successful plot creation
successful_plots = 0
failed_plots = 0

for j_idx, j in enumerate(trig_times):
    try:
        event_time = ostart + j
        logger.info(f"Processing event {j_idx+1}/{len(trig_times)} at {event_time}")
        
        # Get waveform data with a window around the event time
        try:
            st = clw.get_waveforms(network="XG", station="UP1", location="", channel="??Z", 
                                   starttime=(event_time-1), endtime=event_time+10)
            ar = cl.get_waveforms(network="XG", station="UP1", location="", channel="ZG?", 
                                  starttime=(event_time-1), endtime=event_time+10)
        except Exception as data_error:
            logger.error(f"Failed to get waveform data for event at {event_time}: {data_error}")
            failed_plots += 1
            continue
        
        # Check if valid data
        if len(st) == 0 or len(ar) == 0:
            logger.warning(f"No data found for event at {event_time}, skipping")
            failed_plots += 1
            continue
            
        # Check for required channels
        ar_channels = [tr.stats.channel for tr in ar]
        if "ZGC" not in ar_channels or "ZGA" not in ar_channels or "ZGS" not in ar_channels:
            logger.warning(f"Missing required array channels for event at {event_time}, skipping. Available: {ar_channels}")
            failed_plots += 1
            continue
            
        # Check if there's a valid vertical component
        if not st.select(component="Z"):
            logger.warning(f"No vertical component found for event at {event_time}, skipping")
            failed_plots += 1
            continue
            
        # Process waveforms
        st.detrend("linear")
        st.taper(type='cosine', max_percentage=0.05)
        st.filter("bandpass", freqmin=1, freqmax=20)

        # Extract data
        if ar.select(channel="ZGC"):
            rel_power = ar.select(channel="ZGC")[0].data
            all_rp.append(rel_power)
        else:
            logger.warning(f"Missing ZGC channel for event at {event_time}")
            failed_plots += 1
            continue
            
        if ar.select(channel="ZGS"):
            baz = ar.select(channel="ZGS")[0].data
            all_baz.append(baz)
        else:
            logger.warning(f"Missing ZGS channel for event at {event_time}")
            failed_plots += 1
            continue
            
        if ar.select(channel="ZGA"):
            slow = ar.select(channel="ZGA")[0].data
            all_slow.append(slow)
        else:
            logger.warning(f"Missing ZGA channel for event at {event_time}")
            failed_plots += 1
            continue
            
        # Calculate incidence angles based on slowness values
        incidence_angles = []
        wave_types = []
        
        for s in slow:
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
        
        # Convert to numpy arrays
        incidence_angles = np.array(incidence_angles)
        wave_types = np.array(wave_types)
        
        # Close any existing figures
        plt.close('all')
        
        # Create figure with larger size to accommodate bigger fonts
        fig = plt.figure(figsize=(14, 18), dpi=100)
        fig.suptitle(f"Seismic Event Analysis - {event_time.strftime('%Y-%m-%d %H:%M:%S')}", 
                    fontsize=22, fontweight='bold', y=0.98)
        
        # Create a grid layout with 3 rows and 2 columns
        gs = plt.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1], width_ratios=[1, 1],
                          hspace=0.4, wspace=0.4)
        
        common_time_limits = None
        if st.select(component="Z"):
            vert_tr = st.select(component="Z")[0]
            time_data = vert_tr.times("matplotlib")
            
            # Get time limits for alignment
            end_time = max(time_data)
            start_time = min(time_data)
            
            # Common time limits for all time-based plots
            common_time_limits = [start_time, end_time]
            
            # Create the time locator for all plots
            seconds_locator = mdates.SecondLocator(interval=1)
            seconds_formatter = mdates.DateFormatter('%H:%M:%S')
        else:
            logger.warning("No vertical component data for establishing time limits")
            failed_plots += 1
            continue
        
        # PLOT 1: Seismogram - Row 1, Col 1 (Upper Left)
        axtrace = fig.add_subplot(gs[0, 0])
        
        # Plot the vertical component data
        if st.select(component="Z"):
            axtrace.plot(time_data, vert_tr.data, 'k', linewidth=1.2)
            axtrace.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
            axtrace.set_ylabel('Counts', fontsize=16, fontweight='bold')
            axtrace.set_title('a) Vertical Component Waveform', fontsize=18, fontweight='bold', pad=15)
            
            # Set x-axis ticks every second
            axtrace.xaxis.set_major_locator(seconds_locator)
            axtrace.xaxis.set_major_formatter(seconds_formatter)
            
            # Set time limits
            axtrace.set_xlim(common_time_limits)
            
            # Enhance grid for seismogram
            axtrace.grid(True, which='both', axis='x', color='gray', alpha=0.5, linestyle='-')
            axtrace.grid(True, which='major', axis='y', color='gray', alpha=0.5, linestyle='-')
            
            # Rotate time labels
            plt.setp(axtrace.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=13)
            plt.setp(axtrace.yaxis.get_majorticklabels(), fontsize=13)
        else:
            logger.warning("No vertical component data for trace plot")
            failed_plots += 1
            continue
            
        # PLOT 2: Polar Plot - Row 1, Col 2 (Upper Right)
        polar_ax = fig.add_subplot(gs[0, 1], projection='polar')
        
        # Check for the necessary data
        if len(baz) > 0 and len(slow) > 0 and len(rel_power) > 0:
            # Convert backazimuth to radians
            baz_rad = np.radians(baz)
            baz_rad[baz_rad < 0] += 2*np.pi
            baz_rad[baz_rad > 2*np.pi] -= 2*np.pi
            
            # Create 2D histogram for polar plot
            N = int(360./5.)  # 5-degree bins
            abins = np.arange(N + 1) * 2*np.pi / N
            sbins = np.linspace(0, 0.4, 20) 
            
            hist, baz_edges, sl_edges = np.histogram2d(baz_rad, slow, bins=[abins, sbins], weights=rel_power)
            
            # Create meshgrid for pcolormesh
            A, S = np.meshgrid(abins, sbins)

            polar_ax.set_theta_zero_location("N")
            polar_ax.set_theta_direction(-1)
            
            # Use pcolormesh for polar plot with the global colormap
            pcm = polar_ax.pcolormesh(A, S, hist.T, cmap=global_cmap, alpha=0.7, shading='auto')
            
            # Improve polar plot settings
            polar_ax.grid(True, linewidth=1.5)
            
            # Add radial labels
            polar_ax.set_rticks([0.1, 0.2, 0.3, 0.4])
            polar_ax.set_rlabel_position(135)
            polar_ax.set_rmax(0.4)
            polar_ax.set_title('b) Polar Plot: Backazimuth vs. Slowness', fontsize=18, fontweight='bold', pad=20)
            
            # Increase tick label sizes for polar plot
            polar_ax.tick_params(axis='x', labelsize=13)
            polar_ax.tick_params(axis='y', labelsize=13)
        else:
            logger.warning("Missing data for polar plot")
            
        # PLOT 3: Spectrogram - Row 2, Col 1 (Middle Left)
        axspec = fig.add_subplot(gs[1, 0])

        # Get the vertical component data for spectrogram
        if st.select(component="Z"):
            tr = st.select(component="Z")[0]
    
            try:
                # Calculate spectrogram
                specgram = tr.spectrogram(wlen=0.5, per_lap=0.9, show=False, axes=axspec)
        
                # Limit frequency range
                axspec.set_ylim(1, 25)  # Limit frequency to 1-25 Hz
        
                # Set labels and grid
                axspec.set_ylabel('Frequency [Hz]', fontsize=16, fontweight='bold')
                
                # Clear the current x-axis labels and ticks
                axspec.set_xticklabels([])
                axspec.set_xticks([])
                
                # Create secondary axis that matches seismogram time
                ax2 = axspec.twiny()
                ax2.set_xlim(common_time_limits)
                ax2.xaxis.set_major_locator(seconds_locator)
                ax2.xaxis.set_major_formatter(seconds_formatter)
                ax2.xaxis.tick_bottom()
                ax2.xaxis.set_label_position('bottom')
                ax2.tick_params(axis='x', pad=10, labelsize=13)
                
                # Rotate time labels
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=13)
                plt.setp(axspec.yaxis.get_majorticklabels(), fontsize=13)
                
                # Add horizontal grid lines
                axspec.grid(True, which='major', axis='y', color='gray', alpha=0.01, linestyle='-')
                
                # Add vertical grid lines
                for pos in ax2.get_xticks():
                    axspec.axvline(pos, color='gray', alpha=0.01, linestyle='-')
                
                # Set title above the plot
                axspec.set_title('c) Spectrogram', fontsize=18, fontweight='bold', pad=15)
                
            except Exception as e:
                logger.error(f"Error creating spectrogram: {e}")
        else:
            logger.warning("No vertical component data for spectrogram")
            
        # PLOT 4: Incidence Angle - Row 2, Col 2 (Middle Right)
        axangle = fig.add_subplot(gs[1, 1])
        
        # Only plot valid incidence angles
        valid_mask = ~np.isnan(incidence_angles)
        if any(valid_mask):
            # Get matplotlib times for the incidence angle data
            angle_times = ar.select(channel="ZGA")[0].times("matplotlib")
            
            # Calculate point sizes based on relative power if not already defined
            rel_power_norm = rel_power / np.max(rel_power) if np.max(rel_power) > 0 else np.zeros_like(rel_power)
            sizes = 20 + 100 * rel_power_norm
            
            # Use the same marker sizing and coloring scheme as the other plots
            scatter_angle = axangle.scatter(
                angle_times[valid_mask], 
                incidence_angles[valid_mask],
                c=rel_power[valid_mask], 
                cmap=global_cmap, 
                s=sizes[valid_mask], 
                alpha=0.7
            )
            
            # Add markers to indicate wave type
            p_mask = np.logical_and(valid_mask, np.array(wave_types) == "P")
            if any(p_mask):
                axangle.scatter(
                    angle_times[p_mask], 
                    incidence_angles[p_mask],
                    s=30, alpha=0.7, facecolors='none', edgecolors='blue',
                    linewidth=1.5, marker='o', label='P-wave'
                )
                
            s_mask = np.logical_and(valid_mask, np.array(wave_types) == "S")
            if any(s_mask):
                axangle.scatter(
                    angle_times[s_mask], 
                    incidence_angles[s_mask],
                    s=30, alpha=0.7, facecolors='none', edgecolors='red',
                    linewidth=1.5, marker='s', label='S-wave'
                )
            
            axangle.set_ylabel('Incidence Angle [deg]', fontsize=16, fontweight='bold')
            axangle.set_title('d) Incidence Angle vs. Time', fontsize=18, fontweight='bold', pad=15)
            
            # Set x-axis ticks every second
            axangle.xaxis.set_major_locator(seconds_locator)
            axangle.xaxis.set_major_formatter(seconds_formatter)
            
            # Set time limits to match seismogram
            axangle.set_xlim(common_time_limits)
            
            # Set reasonable y-limits for the plot
            axangle.set_ylim(0, 90)
            
            # Enhanced grid with lines every second
            axangle.grid(True, which='major', axis='both', color='gray', alpha=0.5, linestyle='-')
            axangle.legend(loc='upper right', fontsize=14)
            
            # Rotate time labels
            plt.setp(axangle.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=13)
            plt.setp(axangle.yaxis.get_majorticklabels(), fontsize=13)
        else:
            logger.warning("No valid incidence angle data for plot")
        
        # PLOT 5: Backazimuth - Row 3, Col 1 (Lower Left)
        axbaz = fig.add_subplot(gs[2, 0])
        
        # Get matplotlib times for the backazimuth data
        if ar.select(channel="ZGS") and len(ar.select(channel="ZGS")[0].data) > 0:
            baz_times = ar.select(channel="ZGS")[0].times("matplotlib")
            
            # Check for matching data lengths
            if len(baz_times) == len(baz) and len(baz) == len(rel_power):
                scatter_baz = axbaz.scatter(baz_times, baz, 
                           c=rel_power, cmap=global_cmap, s=sizes, alpha=0.7)
                axbaz.set_ylabel('Backazimuth [deg]', fontsize=16, fontweight='bold')
                axbaz.set_xlabel('Time (UTC)', fontsize=16, fontweight='bold')
                axbaz.set_ylim(0, 360)
                axbaz.set_yticks([0, 90, 180, 270, 360])
                axbaz.set_title('e) Backazimuth vs. Time', fontsize=18, fontweight='bold', pad=15)
                
                # Set x-axis ticks every second
                axbaz.xaxis.set_major_locator(seconds_locator)
                axbaz.xaxis.set_major_formatter(seconds_formatter)
                
                # Set time limits to match seismogram
                axbaz.set_xlim(common_time_limits)
                
                # Enhanced grid with lines
                axbaz.grid(True, which='major', axis='both', color='gray', alpha=0.5, linestyle='-')
                
                # Rotate time labels
                plt.setp(axbaz.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=13)
                plt.setp(axbaz.yaxis.get_majorticklabels(), fontsize=13)
            else:
                logger.warning(f"Data length mismatch in backazimuth plot")
        else:
            logger.warning("No backazimuth data available for plot")
        
        # PLOT 6: Slowness - Row 3, Col 2 (Lower Right)
        axslow = fig.add_subplot(gs[2, 1])
        
        # Add horizontal line for the threshold
        axslow.axhline(y=slow_threshold, color='r', linestyle='--', alpha=0.7, 
                       label=f'Threshold ({slow_threshold})')
        
        # Get matplotlib times for the slowness data
        if ar.select(channel="ZGA") and len(ar.select(channel="ZGA")[0].data) > 0:
            slow_times = ar.select(channel="ZGA")[0].times("matplotlib")
            
            # Check for matching data lengths
            if len(slow_times) == len(slow) and len(slow) == len(rel_power):
                # Use scatter for slowness, sized by rel_power like backazimuth
                scatter_slow = axslow.scatter(slow_times, slow, 
                            c=rel_power, cmap=global_cmap, s=sizes, alpha=0.7)
                
                axslow.set_ylabel('Slowness [s/km]', fontsize=16, fontweight='bold')
                axslow.set_xlabel('Time (UTC)', fontsize=16, fontweight='bold')
                axslow.set_title('f) Slowness vs. Time', fontsize=18, fontweight='bold', pad=15)
                
                # Set x-axis ticks every second
                axslow.xaxis.set_major_locator(seconds_locator)
                axslow.xaxis.set_major_formatter(seconds_formatter)
                
                # Set time limits to match seismogram
                axslow.set_xlim(common_time_limits)
                
                # Set y-axis limits
                axslow.set_ylim(0, max(1.0, np.max(slow)*1.1))
                
                # Enhanced grid with lines
                axslow.grid(True, which='major', axis='both', color='gray', alpha=0.5, linestyle='-')
                
                axslow.legend(loc='upper right', fontsize=14)
                
                # Rotate time labels
                plt.setp(axslow.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=13)
                plt.setp(axslow.yaxis.get_majorticklabels(), fontsize=13)
            else:
                logger.warning(f"Data length mismatch in slowness plot")
        else:
            logger.warning("No slowness data available for plot")
        
        # Colorbar for all plots using the same colormap
        if 'pcm' in locals():
            cax = fig.add_axes([0.93, 0.3, 0.02, 0.4])  # Position for vertical colorbar
            cbar = fig.colorbar(pcm, cax=cax)
            cbar.set_label('Relative Power', fontsize=14, fontweight='bold')
            cbar.ax.tick_params(labelsize=12)
        
        # Save figure
        fmt = "png"
        filename = f'{save_path}UP1_{event_time.strftime("%Y%m%d_%H%M%S")}_array.{fmt}'
        

        try:
            plt.savefig(filename, format=fmt, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure: {filename}")
            successful_plots += 1
        except Exception as save_error:
            logger.error(f"Failed to save figure {filename}: {save_error}")
            failed_plots += 1
        
        plt.close("all")
        
    except Exception as e:
        logger.error(f"Error processing event at {event_time}: {e}")
        failed_plots += 1
        plt.close("all")  # Make sure to close all figures even in case of error

logger.info(f"Processing complete! Successfully created {successful_plots} plots, {failed_plots} failed")
logger.info(f"Summary: {len(trig_times)} triggers detected, {len(valid_triggers)} written to CSV, {successful_plots} plots created")