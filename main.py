from src.data_generation import generate_data
from src.model_training import train_model

def main():
    print("1. Генерация данных...")
    generate_data()

    print("\n2. Обучение модели...")
    train_model()

if __name__ == "__main__":
    main()
