import csv
import os

def save_to_csv(data, filename, fieldnames=None):
    """
    Saves a list of dictionaries or a dictionary of lists to a CSV file.
    
    Args:
        data: List of dicts OR dict where each key is a column (list of values).
        filename: Path to the CSV file.
        fieldnames: Optional list of column names.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # If data is a dict of lists, convert it to a list of dicts
    if isinstance(data, dict):
        keys = list(data.keys())
        # Ensure all lists have the same length
        lengths = [len(data[k]) for k in keys]
        if not lengths:
            print(f"Warning: No data to save to {filename}")
            return
        
        min_len = min(lengths)
        if len(set(lengths)) > 1:
            print(f"Warning: Columns in {filename} have different lengths. Truncating to {min_len}.")
        
        list_data = []
        for i in range(min_len):
            row = {k: data[k][i] for k in keys}
            list_data.append(row)
        data = list_data
        if not fieldnames:
            fieldnames = keys

    if not data:
        print(f"Warning: No data to save to {filename}")
        return

    if not fieldnames:
        fieldnames = list(data[0].keys())

    try:
        with open(filename, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"Successfully saved data to {filename}")
    except Exception as e:
        print(f"Error saving to CSV: {e}")
