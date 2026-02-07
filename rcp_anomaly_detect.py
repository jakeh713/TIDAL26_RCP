import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def parse_clock_position(clock_val):
    """Parses 'hh:mm' clock string or datetime.time object into a float (e.g., '3:30' -> 3.5)."""
    if pd.isna(clock_val):
        return np.nan
    
    # If it's a datetime.time object, convert it to a 'HH:MM' string
    if isinstance(clock_val, pd.Timestamp) or hasattr(clock_val, 'strftime'): # Check for datetime.time or pandas Timestamp
        clock_str = clock_val.strftime('%H:%M')
    else:
        clock_str = str(clock_val)

    try:
        parts = clock_str.split(':')
        if len(parts) == 2:
            hours = int(parts[0])
            minutes = int(parts[1])
            return hours + (minutes / 60)
        else:
            # If it's not 'hh:mm' format, try direct conversion to float
            return float(clock_str)
    except ValueError:
        return np.nan

def extract_reference_features(df, reference_types):
    """Extracts reference features (e.g., Girth Welds) from a DataFrame."""
    # Ensure 'event' exists and is used for filtering
    if 'event' not in df.columns:
        raise ValueError("DataFrame must contain 'event' column for reference extraction.")
    
    # Filter for known reference types
    refs_df = df[df['event'].isin(reference_types)].copy()
    
    # Ensure 'log dist. [ft]' exists for distance and no duplicate distances for interpolation
    if 'log dist. [ft]' not in refs_df.columns:
        raise ValueError("Reference DataFrame must contain 'log dist. [ft]' column.")
    
    refs_df = refs_df.drop_duplicates(subset=['log dist. [ft]']).sort_values('log dist. [ft]')
    
    # Rename columns to match expected format for alignment
    refs_df = refs_df.rename(columns={'log dist. [ft]': 'distance', 'event': 'feature_type'})
    
    return refs_df[['distance', 'feature_type']]


def load_data(excel_path, run1_sheet, run2_sheet, refs_sheet, reference_feature_types):
    """Loads ILI and reference data from an Excel file."""
    try:
        # Load all sheets
        run1_raw_df = pd.read_excel(excel_path, sheet_name=run1_sheet)
        run2_raw_df = pd.read_excel(excel_path, sheet_name=run2_sheet)
        refs_raw_df = pd.read_excel(excel_path, sheet_name=refs_sheet)
        
        print("Data loaded successfully from Excel sheets.")

        # --- Standardize Column Names and Data Types ---
        # Map various input column names to a consistent internal name
        common_columns = {
            'Log Dist. [ft]': 'distance', # For 2015/2022 sheets
            'ILI Wheel Count \n[ft.]': 'distance', # For 2022 sheet, if different from Log Dist.
            'log dist. [ft]': 'distance', # For 2007 sheet
            'O\'clock\n[hh:mm]': 'clock_position', # Handle potential newline in column name for 2015/2022
            'O\'clock [hh:mm]': 'clock_position', # Handle absence of newline for 2015/2022
            'O\'clock': 'clock_position', # For 2015 sheet
            'o\'clock': 'clock_position', # For 2007 sheet
            'Event Description': 'feature_type', # For 2015/2022 sheets
            'event': 'feature_type', # For 2007 sheet
            'Metal Loss Depth\n[%]': 'depth_percent', # For 2015/2022 sheets
            'Metal Loss Depth [%]': 'depth_percent', # For 2015/2022 sheets
            'depth [%]': 'depth_percent', # For 2007 sheet
            'Length [in]': 'length', # For 2015/2022 sheets
            'length [in]': 'length', # For 2007 sheet
            'Width [in]': 'width', # For 2015/2022 sheets
            'width [in]': 'width', # For 2007 sheet
            'WT [in]': 'wall_thickness', # For 2015/2022 sheets
            't [in]': 'wall_thickness', # For 2007 sheet
        }

        def process_df(df, year_prefix):
            df = df.copy()
            # Rename columns
            df = df.rename(columns=common_columns, errors='ignore') # Use errors='ignore' to prevent KeyError if a column is missing
            
            # Drop rows where 'distance' is NaN, as these cannot be aligned or matched
            if 'distance' not in df.columns:
                # This should ideally not happen if common_columns is comprehensive
                raise ValueError(f"'distance' column not found after renaming in {year_prefix} data. Please check column mappings.")
            else:
                df = df.dropna(subset=['distance'])

            # Generate feature_id if not present (using index)
            if 'feature_id' not in df.columns:
                df['feature_id'] = f'{year_prefix}_' + df.index.astype(str)
            
            # Parse clock position
            if 'clock_position' in df.columns:
                df['clock_position'] = df['clock_position'].apply(parse_clock_position)
            
            # Select and reorder relevant columns
            required_cols = [
                'feature_id', 'distance', 'clock_position', 'feature_type',
                'depth_percent', 'length', 'width', 'wall_thickness'
            ]
            
            # Ensure all required columns exist, fill missing with NaN
            for col in required_cols:
                if col not in df.columns:
                    df[col] = np.nan
            
            return df[required_cols]

        run1_df = process_df(run1_raw_df, '2015')
        run2_df = process_df(run2_raw_df, '2022')
        
        # Extract reference features from the 2007 data
        # Note: refs_raw_df is processed differently because it's only for reference points
        refs_df = extract_reference_features(refs_raw_df, reference_feature_types)

        return run1_df, run2_df, refs_df

    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please ensure the Excel file is in the correct path.")
        return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred during data loading or processing: {e}")
        return None, None, None

def align_distances(run_df, refs_df):
    """
    Aligns anomaly distances to a common reference frame using linear interpolation.
    This creates a 'corrected_distance' column.
    """
    if run_df.empty or refs_df.empty:
        print("Warning: Empty DataFrame passed to align_distances. Skipping alignment.")
        run_df['corrected_distance'] = run_df['distance'] # Just copy original distance
        return run_df

    # We need at least two reference points to interpolate
    if len(refs_df) < 2:
        print("Warning: Less than 2 reference points available for alignment. Skipping alignment and using original distances.")
        run_df['corrected_distance'] = run_df['distance'] # Just copy original distance
        return run_df

    # Ensure the run_df 'distance' column used for interpolation is numeric and without NaNs
    run_distances_for_interp = run_df['distance'].dropna().sort_values().unique()
    if len(run_distances_for_interp) < 2:
        print("Warning: Not enough unique, non-null distances in run data for interpolation. Skipping alignment and using original distances.")
        run_df['corrected_distance'] = run_df['distance']
        return run_df

    # Ensure the reference distances are unique and sorted for interpolation
    refs_df_sorted = refs_df.drop_duplicates(subset=['distance']).sort_values('distance')
    if len(refs_df_sorted) < 2:
        print("Warning: Not enough unique reference distances for interpolation. Skipping alignment and using original distances.")
        run_df['corrected_distance'] = run_df['distance']
        return run_df

    # Create an interpolation function
    # This function maps the tool's reported distance (odometer) to the true reference distance
    # Here, we're assuming the 'distance' in refs_df represents the *true* distance, and
    # the 'distance' in run_df represents the *reported' distance by the tool that needs correcting.
    interpolation_func = interp1d(
        refs_df_sorted['distance'], # The observed 'distance' from the reference points (as recorded by tool)
        refs_df_sorted['distance'], # The *true* distances of those same reference points
        kind='linear', 
        fill_value="extrapolate"
    )

    # Apply the correction to all anomalies in the run
    # Handle NaN values in run_df['distance'] before applying interpolation
    valid_distances = run_df['distance'].dropna()
    run_df.loc[valid_distances.index, 'corrected_distance'] = interpolation_func(valid_distances)
    run_df['corrected_distance'] = run_df['corrected_distance'].fillna(run_df['distance']) # Fill NaNs with original distance if no correction was applied
    
    return run_df

def find_matches(run1_df, run2_df, distance_tolerance, clock_tolerance):
    """
    Matches anomalies between two runs based on corrected distance, clock position, and dimensions.
    """
    # Prepare dataframes for merging
    run1_df_prefixed = run1_df.add_prefix('r1_')
    run2_df_prefixed = run2_df.add_prefix('r2_')
    
    # Create a temporary key for a rolling join
    # Ensure 'corrected_distance' exists and is numeric for sorting
    run1_df_prefixed['r1_corrected_distance'] = pd.to_numeric(run1_df_prefixed['r1_corrected_distance'], errors='coerce')
    run2_df_prefixed['r2_corrected_distance'] = pd.to_numeric(run2_df_prefixed['r2_corrected_distance'], errors='coerce')

    # Drop NaNs from join keys before sorting for merge_asof
    run1_df_prefixed = run1_df_prefixed.dropna(subset=['r1_corrected_distance']).sort_values('r1_corrected_distance')
    run2_df_prefixed = run2_df_prefixed.dropna(subset=['r2_corrected_distance']).sort_values('r2_corrected_distance')

    # Use pandas merge_asof for a rolling join on the nearest distance
    merged_potential_matches = pd.merge_asof(
        left=run2_df_prefixed,
        right=run1_df_prefixed,
        left_on='r2_corrected_distance',
        right_on='r1_corrected_distance',
        direction='nearest',
        tolerance=distance_tolerance
    )

    # Initialize final_matches to an empty DataFrame to prevent UnboundLocalError
    final_matches = pd.DataFrame() 

    # Filter out bad matches based on clock position and feature type
    # Only consider potential matches where both r1_feature_id and r2_feature_id exist
    merged_potential_matches = merged_potential_matches.dropna(subset=['r1_feature_id', 'r2_feature_id'])

    if merged_potential_matches.empty:
        # If no potential matches were found after dropping NaNs, final_matches remains empty
        pass
    else:
        merged_potential_matches['clock_diff'] = abs(merged_potential_matches['r2_clock_position'] - merged_potential_matches['r1_clock_position'])
        
        # Handle clock position wraparound (e.g., 1 vs 12 o'clock)
        merged_potential_matches['clock_diff'] = merged_potential_matches['clock_diff'].apply(lambda x: min(x, 12 - x) if pd.notna(x) else np.nan)

        # Define compatible feature types (more robustly for new data)
        def are_features_compatible(f1, f2):
            if pd.isna(f1) or pd.isna(f2):
                return False
            f1_lower = str(f1).lower()
            f2_lower = str(f2).lower()
            
            # Specific handling for "metal loss" and "dent" (main anomaly types)
            is_f1_ml = 'metal loss' in f1_lower
            is_f2_ml = 'metal loss' in f2_lower
            is_f1_dent = 'dent' in f1_lower
            is_f2_dent = 'dent' in f2_lower

            if (is_f1_ml and is_f2_ml) or (is_f1_dent and is_f2_dent):
                return True
            
            # Allow "corrosion" to match "metal loss"
            if ('corrosion' in f1_lower and is_f2_ml) or (is_f1_ml and 'corrosion' in f2_lower):
                return True
            
            # Other feature types (like 'Bend', 'AGM', 'Valve') are typically not "anomalies" for growth
            # but if they appear in both and are identical, they could be considered
            if f1_lower == f2_lower and f1_lower not in ['girthweld', 'valve', 'tap', 'tee', 'bend', 'agm', 'magnet', 'area start launcher', 'area end launch trap', 'area end receiver', 'area start tee', 'area end tee', 'area start installation', 'area end installation', 'attachment', 'area start composite wrap', 'area end composite wrap', 'area start sleeve', 'area end sleeve', 'area start casing', 'area end casing', 'flange']:
                return True

            return False

        # Apply filters
        is_clock_match = merged_potential_matches['clock_diff'] <= clock_tolerance
        are_types_compatible_series = merged_potential_matches.apply(
            lambda row: are_features_compatible(row['r1_feature_type'], row['r2_feature_type']), axis=1
        )

        final_matches = merged_potential_matches[is_clock_match & are_types_compatible_series].copy()
        
        # Determine match status
        final_matches['match_status'] = 'MATCH'
    
    # Identify new anomalies (in run 2 but not matched)
    matched_r2_ids = set(final_matches['r2_feature_id'].dropna())
    new_anomalies_df = run2_df[~run2_df['feature_id'].isin(matched_r2_ids)].copy()
    new_anomalies_df['match_status'] = 'NEW'

    # Identify missing anomalies (in run 1 but not matched)
    matched_r1_ids = set(final_matches['r1_feature_id'].dropna())
    missing_anomalies_df = run1_df[~run1_df['feature_id'].isin(matched_r1_ids)].copy()
    missing_anomalies_df['match_status'] = 'MISSING'

    # Prepare new/missing anomalies for concatenation
    # Add 'r1_' or 'r2_' prefix to their own columns for consistent concatenation
    new_anomalies_df = new_anomalies_df.add_prefix('r2_')
    missing_anomalies_df = missing_anomalies_df.add_prefix('r1_')

    # Fill missing columns with NaN to allow concatenation
    all_cols = pd.Index(list(final_matches.columns) + list(new_anomalies_df.columns) + list(missing_anomalies_df.columns)).unique()
    new_anomalies_df = new_anomalies_df.reindex(columns=all_cols)
    missing_anomalies_df = missing_anomalies_df.reindex(columns=all_cols)

    # Combine all results into one dataframe
    result_df = pd.concat([final_matches, new_anomalies_df, missing_anomalies_df], ignore_index=True)
    
    # Drop join_key columns from the final output
    result_df = result_df.drop(columns=['join_key'], errors='ignore')

    return result_df

def calculate_growth(matched_df, years_between_inspections):
    """Calculates growth rates for matched anomalies."""
    if years_between_inspections <= 0:
        return matched_df

    # Calculate growth for matched features only
    growth_mask = matched_df['match_status'] == 'MATCH'
    
    # Depth
    # Ensure columns exist and are numeric before calculation
    matched_df.loc[growth_mask, 'r1_depth_percent'] = pd.to_numeric(matched_df.loc[growth_mask, 'r1_depth_percent'], errors='coerce')
    matched_df.loc[growth_mask, 'r2_depth_percent'] = pd.to_numeric(matched_df.loc[growth_mask, 'r2_depth_percent'], errors='coerce')
    depth_growth = (matched_df.loc[growth_mask, 'r2_depth_percent'] - matched_df.loc[growth_mask, 'r1_depth_percent'])
    matched_df.loc[growth_mask, 'depth_growth_rate_ppy'] = depth_growth / years_between_inspections
    
    # Length
    matched_df.loc[growth_mask, 'r1_length'] = pd.to_numeric(matched_df.loc[growth_mask, 'r1_length'], errors='coerce')
    matched_df.loc[growth_mask, 'r2_length'] = pd.to_numeric(matched_df.loc[growth_mask, 'r2_length'], errors='coerce')
    length_growth = (matched_df.loc[growth_mask, 'r2_length'] - matched_df.loc[growth_mask, 'r1_length'])
    matched_df.loc[growth_mask, 'length_growth_rate_ipy'] = length_growth / years_between_inspections

    # Width
    matched_df.loc[growth_mask, 'r1_width'] = pd.to_numeric(matched_df.loc[growth_mask, 'r1_width'], errors='coerce')
    matched_df.loc[growth_mask, 'r2_width'] = pd.to_numeric(matched_df.loc[growth_mask, 'r2_width'], errors='coerce')
    width_growth = (matched_df.loc[growth_mask, 'r2_width'] - matched_df.loc[growth_mask, 'r1_width'])
    matched_df.loc[growth_mask, 'width_growth_rate_ipy'] = width_growth / years_between_inspections
    
    return matched_df

def main():
    """Main function to run the pipeline alignment and analysis."""
    # --- Configuration ---
    excel_file_path = 'ILIDataV2.xlsx'
    run1_sheet_name = '2015'
    run2_sheet_name = '2022'
    refs_sheet_name = '2007'
    output_path = 'matched_anomalies_excel_output.csv'
    
    # Tolerances for matching
    DISTANCE_TOLERANCE = 5.0  # feet
    CLOCK_TOLERANCE = 1.5   # o'clock (e.g., 1.5 means +/- 1.5 hours)
    YEARS_BETWEEN_RUNS = 7  # 2022 - 2015 = 7 years

    # Define feature types that are considered fixed reference points
    # These will be extracted from the 'refs_sheet_name'
    REFERENCE_FEATURE_TYPES = [
        'GirthWeld', 'Valve', 'Tee', 'AGM', 'Area Start Launcher', 'Area End Launch Trap',
        'Area Start Receive Trap', 'Area End Receiver', 'Area Start Casing', 'Area End Casing',
        'Area Start Composite Wrap', 'Area End Composite Wrap', 'Area Start Sleeve', 'Area End Sleeve',
        'Flange', 'Magnet', 'Tap', 'Bend' # Bends are also fixed, but might also be anomalies? Adjust as needed.
    ]

    # --- 1. Ingest Data ---
    print(f"Loading data from {excel_file_path}...")
    run1_df, run2_df, refs_df = load_data(
        excel_file_path, run1_sheet_name, run2_sheet_name, refs_sheet_name,
        REFERENCE_FEATURE_TYPES
    )
    
    if run1_df is None or run2_df is None or refs_df is None:
        print("Failed to load data. Exiting.")
        return

    print("\nRun 1 Data (Processed Head):")
    print(run1_df.head())
    print("\nRun 2 Data (Processed Head):")
    print(run2_df.head())
    print("\nReference Features (Processed Head):")
    print(refs_df.head())

    # --- 2. Align Distances ---
    print("\nAligning distances for both runs...")
    # The align_distances function assumes the refs_df contains the true distances for matching points
    # For this specific scenario, refs_df is derived from the '2007' sheet.
    # The 'distance' column in refs_df should represent the ground truth for those features.
    
    # We need a set of shared reference points between the ILI run and the ground truth (refs_df)
    # For now, let's assume `refs_df` directly provides the ground truth mapping for alignment.
    # A more robust solution might find common Girth Welds in both run_df and refs_df.
    run1_aligned = align_distances(run1_df.copy(), refs_df)
    run2_aligned = align_distances(run2_df.copy(), refs_df)
    print("Alignment complete.")
    
    print("\nSample of Run 1 with corrected distances:")
    print(run1_aligned[['feature_id', 'distance', 'corrected_distance']].head())

    # --- 3. Match Anomalies ---
    print("\nMatching anomalies between runs...")
    matched_results = find_matches(run1_aligned, run2_aligned, DISTANCE_TOLERANCE, CLOCK_TOLERANCE)
    print("Matching complete.")

    # --- 4. Calculate Growth ---
    print("\nCalculating growth rates...")
    final_results = calculate_growth(matched_results, YEARS_BETWEEN_RUNS)
    print("Growth calculation complete.")

    # --- 5. Output Results ---
    print(f"\nSaving final results to {output_path}...")
    
    # Define columns to show in the output. Filter for existing columns.
    # Adjusted to include potentially new columns from processing and remove temporary ones.
    cols_to_show_initial = [
        'match_status',
        'r1_feature_id', 'r2_feature_id',
        'r1_distance', 'r2_distance', # Original distances
        'r1_corrected_distance', 'r2_corrected_distance',
        'r1_clock_position', 'r2_clock_position',
        'r1_feature_type', 'r2_feature_type',
        'r1_depth_percent', 'r2_depth_percent',
        'r1_length', 'r2_length',
        'r1_width', 'r2_width',
        'r1_wall_thickness', 'r2_wall_thickness',
        'depth_growth_rate_ppy', 'length_growth_rate_ipy', 'width_growth_rate_ipy'
    ]
    
    final_cols = [col for col in cols_to_show_initial if col in final_results.columns]

    final_results.to_csv(output_path, index=False, columns=final_cols)
    print("Process finished successfully.")
    print(f"Results saved to {output_path}")
    print("\nFinal Matched Data Preview:")
    print(final_results[final_cols].head())

if __name__ == '__main__':
    main()