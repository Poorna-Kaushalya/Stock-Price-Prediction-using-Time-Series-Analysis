from stock_model_utils import load_data, fetch_new_data, save_data, preprocess_and_train

def main():
    data = load_data()
    updated_data = fetch_new_data(data)
    save_data(updated_data)
    preprocess_and_train(updated_data)

if __name__ == "__main__":
    main()
