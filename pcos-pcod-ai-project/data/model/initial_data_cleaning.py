import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

def understand_data_structure(df, file_path):
    """
    Performs Step 1: Understand the Data Structure.
    Prints the basic structure, data types, and a preview of the data.
    """
    print("--- Step 1: Understanding Data Structure ---")
    print(f"Successfully loaded dataset from: {file_path}")
    print(f"\n1. Shape of the dataset (rows, columns): {df.shape}")
    
    print("\n2. Data types of each column:")
    # Using df.info() gives a concise summary of dtypes and non-null counts
    df.info()
    
    print("\n3. First 5 rows of the dataset:")
    print(df.head())
    print("-" * 40 + "\n")


def handle_missing_data(df):
    """
    Performs Step 2: Handle Missing Data.
    Detects missing values and applies an imputation strategy.
    """
    print("--- Step 2: Handling Missing Data ---")
    
    # 1. Detect missing values
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0] # Filter to show only columns with missing data
    
    if not missing_values.empty:
        print("1. Detected missing values (column: count):")
        print(missing_values)
    else:
        print("1. No missing values detected in the dataset.")
        print("-" * 40 + "\n")
        return df

    # Create a copy to avoid modifying the original DataFrame in place
    df_cleaned = df.copy()

    # 2. Use imputation to fill missing values
    print("\n2. Applying imputation strategy:")
    for col in df_cleaned.columns:
        if df_cleaned[col].isnull().any():
            # For numeric columns, fill with the median
            if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                median_val = df_cleaned[col].median()
                df_cleaned[col].fillna(median_val, inplace=True)
                print(f"  - Filled missing values in numeric column '{col}' with median ({median_val:.2f})")
            # For categorical/object columns, fill with the mode
            else:
                mode_val = df_cleaned[col].mode()[0]
                df_cleaned[col].fillna(mode_val, inplace=True)
                print(f"  - Filled missing values in categorical column '{col}' with mode ('{mode_val}')")

    # 3. Verify that no missing values remain
    print("\n3. Verifying missing value counts after imputation:")
    remaining_missing = df_cleaned.isnull().sum().sum()
    if remaining_missing == 0:
        print("  - Success! No missing values remain.")
    else:
        print(f"  - Warning! {remaining_missing} missing values still exist.")
        
    print("-" * 40 + "\n")
    return df_cleaned


def clean_and_normalize_text(df, column_name):
    """
    Performs Step 3: Clean and Normalize Text for a specific column.
    - Converts text to lowercase.
    - Removes URLs, special characters, and extra whitespace.
    """
    print(f"--- Step 3: Cleaning and Normalizing Text for column '{column_name}' ---")

    if column_name not in df.columns:
        print(f"  - Warning: Column '{column_name}' not found in the DataFrame. Skipping.")
        print("-" * 40 + "\n")
        return df

    if not pd.api.types.is_string_dtype(df[column_name]):
        print(f"  - Warning: Column '{column_name}' is not a text/string type. Skipping.")
        print("-" * 40 + "\n")
        return df

    df_cleaned = df.copy()
    
    print(f"1. Original first 5 values in '{column_name}':\n{df_cleaned[column_name].head().to_string(index=False)}")

    # Apply cleaning steps
    # 1. Lowercase
    df_cleaned[column_name] = df_cleaned[column_name].str.lower()
    # 2. Remove URLs
    df_cleaned[column_name] = df_cleaned[column_name].str.replace(r'http\S+|www\S+', '', case=False, regex=True)
    # 3. Remove non-alphanumeric characters (except spaces)
    df_cleaned[column_name] = df_cleaned[column_name].str.replace(r'[^a-z0-9\s]', '', regex=True)
    # 4. Remove extra whitespace
    df_cleaned[column_name] = df_cleaned[column_name].str.strip().str.replace(r'\s+', ' ', regex=True)

    print(f"\n2. Cleaned first 5 values in '{column_name}':\n{df_cleaned[column_name].head().to_string(index=False)}")
    print("-" * 40 + "\n")
    return df_cleaned


def process_datetime_column(df, column_name):
    """
    Performs Step 4: Process Date and Time columns.
    - Converts column to datetime objects.
    - Extracts time components.
    - Sorts the DataFrame by the date column.
    """
    print(f"--- Step 4: Processing Datetime column '{column_name}' ---")

    if column_name not in df.columns:
        print(f"  - Warning: Column '{column_name}' not found. Skipping datetime processing.")
        print("-" * 40 + "\n")
        return df

    df_processed = df.copy()

    # 1. Convert to datetime, coercing errors to NaT (Not a Time)
    df_processed[column_name] = pd.to_datetime(df_processed[column_name], errors='coerce')
    print(f"1. Converted '{column_name}' to datetime objects.")

    # 2. Extract time components into new columns
    df_processed[f'{column_name}_year'] = df_processed[column_name].dt.year
    df_processed[f'{column_name}_month'] = df_processed[column_name].dt.month
    df_processed[f'{column_name}_day'] = df_processed[column_name].dt.day
    print("2. Extracted 'year', 'month', and 'day' into new columns.")

    # 3. Sort chronologically
    df_processed.sort_values(by=column_name, inplace=True)
    print(f"3. Sorted DataFrame chronologically by '{column_name}'.")

    print("-" * 40 + "\n")
    return df_processed


def handle_outliers_iqr(df, columns):
    """
    Performs Step 5: Handle Outliers using the IQR method.
    Detects outliers in specified numeric columns and caps them.
    """
    print("--- Step 5: Handling Outliers using IQR ---")
    
    df_processed = df.copy()
    
    for col in columns:
        if col not in df_processed.columns or not pd.api.types.is_numeric_dtype(df_processed[col]):
            print(f"  - Skipping non-numeric or non-existent column: '{col}'")
            continue
            
        print(f"\nProcessing column: '{col}'")
        
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        print(f"  - Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
        print(f"  - Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")
        
        # Identify outliers
        outliers_low = df_processed[col] < lower_bound
        outliers_high = df_processed[col] > upper_bound
        num_outliers = outliers_low.sum() + outliers_high.sum()
        
        if num_outliers > 0:
            print(f"  - Found {num_outliers} outliers. Capping them.")
            
            # Cap the outliers
            df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
        else:
            print("  - No outliers found.")
            
    print("-" * 40 + "\n")
    return df_processed


def encode_categorical_data(df, columns):
    """
    Performs Step 6: Encode Categorical Data using one-hot encoding.
    """
    print("--- Step 6: Encoding Categorical Data ---")
    
    df_encoded = df.copy()
    
    # Identify columns that exist in the dataframe
    columns_to_encode = [col for col in columns if col in df_encoded.columns]
    if not columns_to_encode:
        print("  - No specified categorical columns found in the DataFrame. Skipping.")
        print("-" * 40 + "\n")
        return df

    print(f"1. Applying one-hot encoding to: {columns_to_encode}")
    
    # Use pandas get_dummies for one-hot encoding
    df_encoded = pd.get_dummies(df_encoded, columns=columns_to_encode, prefix=columns_to_encode, drop_first=True)
    
    print("\n2. DataFrame shape after encoding:", df_encoded.shape)
    print("3. New columns created (sample):", [col for col in df_encoded.columns if any(cat_col in col for cat_col in columns_to_encode)][:5])
    
    print("-" * 40 + "\n")
    return df_encoded


def scale_numeric_data(df):
    """
    Performs Step 7: Scale and Normalize Numeric Data using Standardization.
    """
    print("--- Step 7: Scaling Numeric Data ---")
    
    df_scaled = df.copy()
    
    # Identify numeric columns to scale.
    # We exclude binary-like columns (only 0s and 1s) which are often outcomes of encoding
    # and don't need scaling. We also exclude the target variable if it's present.
    numeric_cols = df_scaled.select_dtypes(include=np.number).columns.tolist()
    
    # Filter out binary columns and known target variables
    cols_to_scale = [
        col for col in numeric_cols 
        if len(df_scaled[col].unique()) > 2 and 'PCOS' not in col
    ]

    if not cols_to_scale:
        print("  - No numeric columns requiring scaling were found. Skipping.")
        print("-" * 40 + "\n")
        return df

    print(f"1. Applying StandardScaler to {len(cols_to_scale)} numeric columns.")
    
    scaler = StandardScaler()
    df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])
    
    print("2. Numeric data scaled successfully.")
    print("-" * 40 + "\n")
    return df_scaled


def merge_data_sources(df_main, df_secondary, key_column, merge_type='inner'):
    """
    Performs Step 8: Merge Multiple Data Sources.
    """
    print("--- Step 8: Merging Data Sources ---")
    
    if key_column not in df_main.columns or key_column not in df_secondary.columns:
        print(f"  - Error: Key column '{key_column}' not found in one or both DataFrames. Skipping merge.")
        print("-" * 40 + "\n")
        return df_main

    print(f"1. Merging dataframes on key: '{key_column}' using a '{merge_type}' join.")
    print(f"   - Shape before merge: Main={df_main.shape}, Secondary={df_secondary.shape}")

    # Perform the merge
    df_merged = pd.merge(df_main, df_secondary, on=key_column, how=merge_type)
    
    print(f"   - Shape after merge: {df_merged.shape}")

    # Remove duplicates that may arise from merging
    duplicates_before = df_merged.duplicated().sum()
    if duplicates_before > 0:
        print(f"2. Found and removed {duplicates_before} duplicate rows after merging.")
        df_merged.drop_duplicates(inplace=True)
        print(f"   - Shape after removing duplicates: {df_merged.shape}")
    else:
        print("2. No duplicate rows found after merging.")

    print("-" * 40 + "\n")
    return df_merged


def validate_and_save_data(df, output_path):
    """
    Performs Step 10: Final Data Validation and Saving.
    """
    print("--- Step 10: Final Data Validation and Saving ---")

    # 1. Final validation check for missing values
    remaining_missing = df.isnull().sum().sum()
    if remaining_missing == 0:
        print("1. Validation successful: No missing values remain in the dataset.")
    else:
        print(f"1. Validation failed: {remaining_missing} missing values still exist. Please review previous steps.")
        return # Stop if data is not clean

    # 2. Confirm data types (optional, but good practice)
    print("\n2. Final data types summary:")
    df.info()

    # 3. Save the cleaned dataset
    df.to_csv(output_path, index=False)
    print(f"\n3. Successfully saved cleaned dataset to: {output_path}")
    print("-" * 40 + "\n")


def perform_eda(df, output_dir='eda_plots'):
    """
    Performs Step 11 & 12: Exploratory Data Analysis (Univariate Analysis).
    Generates and saves plots for numeric and categorical columns.
    """
    print("--- Step 11 & 12: Performing Exploratory Data Analysis ---")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"1. Created directory for plots: '{output_dir}'")

    # Identify numeric and categorical columns for plotting
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # 2. Plot distributions for numeric columns
    print("\n2. Generating plots for numeric columns...")
    for col in numeric_cols:
        plt.figure(figsize=(12, 6))
        
        # Histogram
        plt.subplot(1, 2, 1)
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'Histogram of {col}')
        
        # Box Plot
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col])
        plt.title(f'Box Plot of {col}')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'numeric_{col.replace(" ", "_").replace("/", "")}.png')
        plt.savefig(plot_path)
        plt.close()
    print(f"  - Saved {len(numeric_cols)} numeric plots to '{output_dir}'.")

    # 3. Plot counts for categorical columns
    print("\n3. Generating plots for categorical columns...")
    for col in categorical_cols:
        plt.figure(figsize=(10, 6))
        sns.countplot(y=df[col], order=df[col].value_counts().index)
        plt.title(f'Count of {col}')
        plt.xlabel('Count')
        plt.ylabel(col)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'categorical_{col.replace(" ", "_").replace("/", "")}.png')
        plt.savefig(plot_path)
        plt.close()
    print(f"  - Saved {len(categorical_cols)} categorical plots to '{output_dir}'.")

    print("-" * 40 + "\n")


def perform_bivariate_and_correlation_analysis(df, target_variable, output_dir='eda_plots'):
    """
    Performs Steps 13, 15, 18: Bivariate, Correlation, and Target Relationship Analysis.
    """
    print("--- Steps 13, 15, 18: Performing Bivariate and Correlation Analysis ---")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Ensure target variable exists
    if target_variable not in df.columns:
        print(f"  - Target variable '{target_variable}' not found. Skipping analysis.")
        return

    numeric_df = df.select_dtypes(include=np.number)

    # 1. Correlation Heatmap (Step 15)
    print("1. Generating correlation heatmap...")
    plt.figure(figsize=(20, 16))
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
    plt.title('Correlation Matrix of Numeric Features', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()
    print(f"  - Saved correlation heatmap to '{output_dir}'.")

    # 2. Relationship with Target Variable (Step 18)
    # Get top 5 features most correlated with the target
    top_features = correlation_matrix[target_variable].abs().sort_values(ascending=False).index[1:6]

    print("\n2. Generating plots for key features vs. target variable...")
    for col in top_features:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=target_variable, y=col, data=df)
        plt.title(f'{col} vs. {target_variable}', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'bivariate_boxplot_{col.replace(" ", "_")}_vs_target.png'))
        plt.close()
    print(f"  - Saved {len(top_features)} box plots vs. target to '{output_dir}'.")

    # 3. Pair Plot for Bivariate Analysis (Step 13)
    # Using top features plus target for a readable plot
    pairplot_cols = top_features.tolist() + [target_variable]
    print("\n3. Generating pair plot for top features...")
    
    # Check if there are columns to plot
    if len(pairplot_cols) > 1:
        pair_plot = sns.pairplot(df[pairplot_cols], hue=target_variable, palette='viridis')
        pair_plot.fig.suptitle('Pair Plot of Top Features vs. PCOS', y=1.02, fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'bivariate_pairplot.png'))
        plt.close()
        print(f"  - Saved pair plot to '{output_dir}'.")
    else:
        print("  - Not enough columns for a pair plot.")

    print("-" * 40 + "\n")


def main():
    """
    Main function to execute the data cleaning steps.
    """
    # Define the path to your dataset
    # NOTE: This script assumes it's run from the 'pcos-pcod-ai-project/data/' directory
    file_path = 'PCOS_infertility.csv'
    output_file_path = 'cleaned_pcos_data.csv'
    
    try:
        df = pd.read_csv(file_path)
        understand_data_structure(df, file_path)
        cleaned_df = handle_missing_data(df)

        # --- Example usage for Step 3 ---
        # Specify the text column you want to clean.
        # If your dataset doesn't have a text column, this step will be skipped.
        text_column_to_clean = 'AMH(ng/mL)' # Example: change to a real text column name if available
        cleaned_df = clean_and_normalize_text(cleaned_df, text_column_to_clean)

        # --- Example usage for Step 4 ---
        # Specify the date column you want to process.
        date_column_to_process = 'Marriage Status (Yrs)' # Example: change to a real date column if available
        cleaned_df = process_datetime_column(cleaned_df, date_column_to_process)

        # --- Example usage for Step 5 ---
        # Specify the numeric columns to check for outliers
        numeric_columns_for_outliers = [
            'Age (yrs)', 'Weight (Kg)', 'Height(Cm)', 'BMI', 'Pulse rate(bpm) ',
            'RR (breaths/min)', 'Hb(g/dl)', 'Cycle(R/I)', 'Cycle length(days)',
            'No. of aborptions'
        ]
        cleaned_df = handle_outliers_iqr(cleaned_df, numeric_columns_for_outliers)

        # --- Step 11 & 12: Perform Exploratory Data Analysis ---
        perform_eda(cleaned_df.copy())

        # --- Steps 13, 15, 18: Bivariate and Correlation Analysis ---
        perform_bivariate_and_correlation_analysis(cleaned_df.copy(), target_variable='PCOS (Y/N)')

        # --- Example usage for Step 6 ---
        # Specify the categorical columns to encode
        categorical_columns_to_encode = [
            'Blood Group', 'Weight gain(Y/N)', 'hair growth(Y/N)',
            'Skin darkening (Y/N)', 'Hair loss(Y/N)', 'Pimples(Y/N)',
            'Fast food (Y/N)', 'Reg.Exercise(Y/N)', 'PCOS (Y/N)',
            'Pregnant(Y/N)'
        ]
        cleaned_df = encode_categorical_data(cleaned_df, categorical_columns_to_encode)

        # --- Step 7: Scale Numeric Data ---
        cleaned_df = scale_numeric_data(cleaned_df)

        # --- Example usage for Step 8 ---
        # This is a placeholder. You would load your second dataset here.
        # For example:
        # secondary_file_path = 'PCOS_data_without_infertility.xlsx'
        # df_secondary = pd.read_excel(secondary_file_path)
        # # Ensure column names are consistent before merging, e.g., 'Patient File No.' vs 'Patient_File_No'
        # merge_key = 'Patient File No.' 
        # cleaned_df = merge_data_sources(cleaned_df, df_secondary, key_column=merge_key)
        print("--- Step 8: Merging Data Sources (Skipped) ---")
        print("  - No secondary dataset provided for merging in this example.")
        print("-" * 40 + "\n")

        # --- Step 10: Final Validation and Saving ---
        validate_and_save_data(cleaned_df, output_file_path)

        print("Full data cleaning pipeline complete.")
        # You can now use 'cleaned_df' for further processing
    except FileNotFoundError:
        print(f"Error: A file was not found. Please check the path: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()