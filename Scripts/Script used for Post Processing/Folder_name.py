import os

def rename_folders(base_folder, user_input):
    try:
        user_input = int(user_input)  # Ensure the user input is an integer
    except ValueError:
        print("Invalid input. Please provide a valid integer.")
        return

    # Sort folders numerically after extracting their indices
    folder_names = sorted(
        [folder for folder in os.listdir(base_folder) if folder.startswith("Run_")],
        key=lambda x: int(x.split("_")[1])
    )

    # First, rename all folders to temporary names to avoid conflicts
    temp_names = {}
    for folder_name in folder_names:
        temp_name = f"{folder_name}_temp"
        old_path = os.path.join(base_folder, folder_name)
        temp_path = os.path.join(base_folder, temp_name)
        os.rename(old_path, temp_path)
        temp_names[temp_name] = folder_name
        print(f"Temporarily renamed {folder_name} to {temp_name}")

    # Then, rename the temporary folders to their final names
    for temp_name, original_name in temp_names.items():
        folder_index = int(original_name.split("_")[1])
        new_index = folder_index + user_input - 1
        new_folder_name = f"Run_{new_index}"
        temp_path = os.path.join(base_folder, temp_name)
        new_path = os.path.join(base_folder, new_folder_name)
        os.rename(temp_path, new_path)
        print(f"Renamed {temp_name} to {new_folder_name}")

    print("Folder renaming completed.")

# Example usage
if __name__ == "__main__":
    base_folder = input("Enter the path to the folder containing Run_ folders: ").strip()
    user_input = input("Enter the starting number: ").strip()
    rename_folders(base_folder, user_input)