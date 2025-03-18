import os
import pandas as pd

def aggregate_multiple_csv_files(base_folder, output_folder):
    # List of expected CSV file names
    csv_files = [
        "captured_emissions.csv", "emissions.csv", "new_capacity.csv", 
        "tech_cost.csv", "tech_production_annual.csv", "tech_production.csv",
        "tech_use_annual.csv", "tech_use.csv", "total_capacity.csv",
        "unmet_demand_annual.csv", "unmet_demand.csv"
    ]

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each file type
    for file_name in csv_files:
        aggregated_data = []  # Reset for each file type

        # Sort folders numerically based on the number after "Run_"
        folder_names = sorted(
            [folder for folder in os.listdir(base_folder) if folder.startswith("Run_")],
            key=lambda x: int(x.split("_")[1])
        )

        for folder_name in folder_names:
            folder_path = os.path.join(base_folder, folder_name)
            csv_path = os.path.join(folder_path, file_name)

            if os.path.exists(csv_path):
                try:
                    # Extract and zero-pad the numeric part of the run number
                    run_number = folder_name.split("_")[1]
                    run_number_padded = f"{int(run_number):02d}"  # Pad with leading zeros

                    df = pd.read_csv(csv_path)
                    if df.empty:
                        print(f"File {csv_path} is empty. Skipping.")
                        continue
                    # Insert the padded run number as the first column
                    df.insert(0, "Run number", run_number_padded)
                    aggregated_data.append(df)
                    print(f"Processed {csv_path} with {len(df)} rows.")
                except Exception as e:
                    print(f"Error reading file {csv_path}: {e}")
            else:
                print(f"File not found: {csv_path}")

        # Save the aggregated data for the current file type
        if aggregated_data:
            result_df = pd.concat(aggregated_data, ignore_index=True)
            output_file_path = os.path.join(output_folder, file_name)
            result_df.to_csv(output_file_path, index=False)
            print(f"Aggregated data for {file_name} saved to {output_file_path}")
        else:
            print(f"No data to aggregate for {file_name}.")

if __name__ == "__main__":
    # Get the base folder path from the user
    base_folder = input("Enter the path to the folder containing Run_ folders: ").strip()
    output_folder = os.path.join(base_folder, "aggregated_results")
    aggregate_multiple_csv_files(base_folder, output_folder)