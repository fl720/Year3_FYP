import random
import csv
from comfitness_level import comfort_level

def generate_data(filename="comfitness_training.csv", num_samples=100):
    # Define the range for temperature and humidity
    temp_range = (-20, 50)
    humidity_range = (0, 100)
    
    # Open the CSV file for writing
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header
        writer.writerow(["Temperature", "Humidity", "Comfort Level"])
        
        # Generate and write the dataset
        for _ in range(num_samples):
            temperature = random.uniform(*temp_range)  # Generate a float temperature
            humidity = random.uniform(*humidity_range)  # Generate a float humidity level
            comfort = comfort_level(temperature, humidity)  # Get comfort level output
            
            writer.writerow([round(temperature, 2), round(humidity, 2), comfort])
    
    print(f"Dataset saved to {filename}")

if __name__ == "__main__":
    generate_data()