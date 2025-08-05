import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from matplotlib.colors import to_rgba
import warnings
warnings.filterwarnings('ignore')

# Improved plotting configuration
plt.style.use('classic')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 32
plt.rcParams['axes.grid'] = True
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.7
plt.rcParams['axes.titlesize'] = 34  
plt.rcParams['axes.labelsize'] = 32  
plt.rcParams['xtick.labelsize'] = 30  
plt.rcParams['ytick.labelsize'] = 30  

output_dir = "./Grenzgletscher_fk/plots"
os.makedirs(output_dir, exist_ok=True)

def clean_numeric_value(value):
    # Clean numeric values that might have periods as thousand separators
    if pd.isna(value) or value == '':
        return np.nan
    
    if isinstance(value, str):
        # Remove periods used as thousand separators, but keep decimal points
        # First, check if it's scientific notation
        if 'E' in value.upper():
            try:
                return float(value.replace(',', '.'))
            except:
                return np.nan
        
        # Count periods to determine if they're thousand separators
        period_count = value.count('.')
        if period_count > 1:
            # Multiple periods likely means thousand separators
            # Keep only the last period as decimal point
            parts = value.split('.')
            if len(parts) > 1:
                integer_part = ''.join(parts[:-1])
                decimal_part = parts[-1]
                cleaned_value = f"{integer_part}.{decimal_part}"
            else:
                cleaned_value = value
        else:
            cleaned_value = value
        
        try:
            return float(cleaned_value)
        except:
            return np.nan
    
    return float(value) if not pd.isna(value) else np.nan

def load_data(filepath):
    print(f"Loading data from {filepath}")
    
    try:
        # Try reading with different separators
        if filepath.endswith('.csv'):
            try:
                df = pd.read_csv(filepath, sep=';')
            except:
                try:
                    df = pd.read_csv(filepath, sep=',')
                except:
                    df = pd.read_csv(filepath, sep='\t')
        else:
            df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    print(f"Initial shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Handle class column naming
    if 'class_name' in df.columns and 'class' not in df.columns:
        df = df.rename(columns={'class_name': 'class'})
    
    # Clean numeric columns
    numeric_columns = []
    for col in df.columns:
        if col not in ['event_time', 'class', 'class_name']:
            try:
                # Try to convert the first few non-null values
                sample_values = df[col].dropna().head(10)
                cleaned_sample = [clean_numeric_value(val) for val in sample_values]
                if any(not pd.isna(val) for val in cleaned_sample):
                    print(f"Cleaning numeric column: {col}")
                    df[col] = df[col].apply(clean_numeric_value)
                    numeric_columns.append(col)
            except Exception as e:
                print(f"Could not convert column {col} to numeric: {e}")
    
    print(f"Successfully converted {len(numeric_columns)} numeric columns")
    
    # Handle time column
    if 'event_time' in df.columns:
        try:
            df['event_time'] = pd.to_datetime(df['event_time'])
            df = df.sort_values('event_time')
            
            first_time = df['event_time'].min()
            df['time_delta_min'] = (df['event_time'] - first_time).dt.total_seconds() / 60
            df['time_delta_sec'] = (df['event_time'] - first_time).dt.total_seconds()
            df['time_delta_days'] = (df['event_time'] - first_time).dt.total_seconds() / (60*60*24)
            
            df['day_of_year'] = df['event_time'].dt.dayofyear
            df['hour_of_day'] = df['event_time'].dt.hour
        except Exception as e:
            print(f"Could not process event_time: {e}")
            df['event_time'] = pd.date_range(start='2024-03-19', periods=len(df), freq='H')
            df['time_delta_min'] = np.arange(len(df))
            df['time_delta_sec'] = np.arange(len(df)) * 60
            df['time_delta_days'] = np.arange(len(df)) / 24
    else:
        df['event_time'] = pd.date_range(start='2024-03-19', periods=len(df), freq='H')
        df['time_delta_min'] = np.arange(len(df))
        df['time_delta_sec'] = np.arange(len(df)) * 60
        df['time_delta_days'] = np.arange(len(df)) / 24
    
    print(f"Final shape: {df.shape}")
    print(f"Classes found: {sorted(df['class'].unique()) if 'class' in df.columns else 'No class column'}")
    print(f"Loaded {len(df)} events with {len(df.columns)} features")
    
    return df

def get_class_colors(classes):
    color_palette = {
        0: 'tab:blue',
        1: 'tab:orange', 
        2: 'tab:green',
        3: 'tab:red',
        4: 'tab:purple',
        5: 'tab:brown',
        6: 'tab:pink',
        7: 'tab:gray',
        8: 'tab:olive',
        9: 'tab:cyan'
    }
    
    # Handle string class names
    unique_classes = sorted(classes)
    if isinstance(unique_classes[0], str):
        color_map = {}
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
                 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        for i, cls in enumerate(unique_classes):
            color_map[cls] = colors[i % len(colors)]
        return color_map
    else:
        return {cls: color_palette[i % 10] for i, cls in enumerate(unique_classes)}

def save_plot(fig, filename):
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {filepath}")

def plot_boxplots_for_all_features(df):
    exclude_cols = ['event_time', 'time_delta_min', 'time_delta_sec', 'time_delta_days', 
                    'day_of_year', 'hour_of_day', 'class', 'class_name']
    feature_cols = [col for col in df.columns 
                   if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    if not feature_cols:
        print("No numeric feature columns found for box plots!")
        return
    
    print(f"Creating box plots for {len(feature_cols)} features")
    
    classes = sorted(df['class'].unique())
    class_colors = get_class_colors(classes)
    
    # Create box plots for each feature across all classes
    for i, col in enumerate(feature_cols):
        print(f"Processing feature {i+1}/{len(feature_cols)}: {col}")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Prepare data for each class (using absolute values)
        class_data = []
        class_labels = []
        for cls in classes:
            data = df[df['class'] == cls][col].dropna()
            if len(data) > 0:
                class_data.append(np.abs(data.values))  # Take absolute values
                # Use class names directly, just capitalize them
                class_labels.append(str(cls).capitalize())
        
        if not class_data:
            print(f"No data found for feature {col}")
            plt.close(fig)
            continue
        
        # Create box plot
        try:
            bplot = ax.boxplot(class_data, 
                      patch_artist=True,
                      notch=False,
                      showfliers=True,
                      widths=0.6,
                      medianprops={'color': 'red', 'linewidth': 2},
                      whiskerprops={'linewidth': 1.2},
                      capprops={'linewidth': 1.2},
                      boxprops={'linewidth': 1.2})
            
            # Color the boxes according to class colors
            for j, box in enumerate(bplot['boxes']):
                if j < len(classes):
                    box_color = class_colors[classes[j]]
                    box.set(facecolor=box_color, alpha=0.7)
            
            # Add scatter points with slight jitter for better distribution visibility
            for j, cls in enumerate(classes):
                if j >= len(class_data):
                    continue
                y = class_data[j]  # Already absolute values from above
                if len(y) > 0:
                    # Limit number of points for readability
                    if len(y) > 500:
                        idx = np.random.choice(len(y), 500, replace=False)
                        y = y[idx]
                    
                    x = np.random.normal(j+1, 0.06, size=len(y))
                    ax.scatter(x, y, alpha=0.3, s=6, c=class_colors[cls], edgecolor='none')
            
            # Increased fontsize for title and labels
            ax.set_title(f'Distribution of |{col}| by Class', fontsize=20, fontweight='bold')
            ax.set_ylabel(f'|{col}|', fontsize=18)
            ax.set_xlabel('Class', fontsize=18)
            ax.set_xticklabels(class_labels, fontsize=16)
            
            # Make tick labels larger
            ax.tick_params(axis='y', labelsize=16)
            ax.tick_params(axis='x', labelsize=16)
            
            ax.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            # Clean filename
            clean_col_name = col.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
            save_plot(fig, f'boxplot_{clean_col_name}.png')
            
        except Exception as e:
            print(f"Error creating box plot for {col}: {e}")
            plt.close(fig)
    
    # Create summary statistics (using absolute values)
    try:
        print("\nCreating summary statistics...")
        # Apply absolute values to feature columns for summary statistics
        abs_df = df.copy()
        for col in feature_cols:
            abs_df[col] = np.abs(df[col])
        
        summary_stats = abs_df[feature_cols + ['class']].groupby('class').agg(['mean', 'std', 'median', 'min', 'max'])
        summary_stats.to_csv(os.path.join(output_dir, 'summary_statistics_by_class_absolute.csv'))
        print("Summary statistics (absolute values) saved to summary_statistics_by_class_absolute.csv")
    except Exception as e:
        print(f"Error creating summary statistics: {e}")

def main():
    # Path to the labeled features CSV file
    input_file = "./Grenzgletscher_fk/known_features_labeled.csv"
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found!")
        return
    
    # Load data
    df = load_data(input_file)
    
    if df is None:
        print("Failed to load data!")
        return
    
    # Check if we have a class column
    if 'class' not in df.columns:
        print("Error: No 'class' column found in the data!")
        return
    
    # Run box plot visualization
    print("Starting box plot creation...")
    plot_boxplots_for_all_features(df)
    
    print("Box plot visualizations completed!")
    print(f"All plots saved to: {output_dir}")

if __name__ == "__main__":
    main()