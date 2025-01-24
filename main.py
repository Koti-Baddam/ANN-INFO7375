import os

def main():
    print("Options:")
    print("1. Generate Dataset")
    print("2. Train Perceptron")
    print("3. Test Perceptron")
    choice = int(input("Select an option: "))

    if choice == 1:
        from data_preparation import create_handwritten_digits
        font_path = "arial.ttf"
        create_handwritten_digits("train_digits", font_path)
        create_handwritten_digits("test_digits", font_path)
        print("Dataset generated.")
    elif choice == 2:
        os.system("python train_perceptron.py")
    elif choice == 3:
        os.system("python test_perceptron.py")
    else:
        print("Invalid option. Exiting.")

if __name__ == "__main__":
    main()
