import pandas as pd


def count_columns(csv_file_path):
    """
    Count the number of columns in a CSV file.
    
    Args:
        csv_file_path (str): Path to the CSV file
        
    Returns:
        int: Number of columns in the CSV file
    """
    df = pd.read_csv(csv_file_path)
    return len(df.columns)


if __name__ == "__main__":
    # Example usage
    file_path = "data/val_with_bow.csv"
    num_columns = count_columns(file_path)
    print(f"Number of columns: {num_columns}")
