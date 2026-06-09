from src.preprocessing import load_and_clean_data
from src.train import train_pipeline

if __name__ == "__main__":
    print("Starting Car Price Prediction Pipeline...")
    data = load_and_clean_data('Data/car data.csv')
    train_pipeline(data)
    print("pipeline completed successfull!")

