import random
import csv

# Generate random numbers and save to CSV
def generate_random_numbers_to_csv(file_path, rows, cols, min_val, max_val):
    """
    Generate a CSV file with random numbers.
    
    Parameters:
        file_path (str): Path to save the CSV file.
        rows (int): Number of rows of random numbers.
        cols (int): Number of columns of random numbers.
        min_val (int): Minimum value for the random numbers.
        max_val (int): Maximum value for the random numbers.
    """
    try:
        with open(file_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            
            # Write header
            header = [f"Column{i+1}" for i in range(cols)]
            writer.writerow(header)
            
            # Write random data
            for _ in range(rows):
                row = [random.randint(min_val, max_val) for _ in range(cols)]
                writer.writerow(row)
        
        print(f"Random numbers saved to {file_path}")
    except Exception as e:
        print(f"Error generating CSV file: {e}")

# Example usage
output_file = 'random_numbers.csv'  # Replace with your desired file path
generate_random_numbers_to_csv(output_file, rows=100000, cols=1, min_val=1, max_val=10000000)
