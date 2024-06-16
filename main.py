from studentperformance import run_student
from newsshares import run_news
from heartDisease import run_heart
from fraudDetection import run_fraud


def get_float(prompt, min_value=0, max_value=1):
    while True:
        try:
            value = float(input(prompt))
            if min_value <= value <= max_value:
                return value
            else:
                print(
                    f"Please enter a value between {min_value} and {max_value}.")
        except ValueError:
            print("Invalid input. Please enter a numerical value.")


def get_input(prompt, valid_options):
    while True:
        value = input(prompt).upper()
        if value in valid_options:
            return value
        else:
            print(
                f"Invalid input. Please enter one of the following: {', '.join(valid_options)}")


print("Welcome to the Machine Learning Model Runner!")
problem_type = get_input(
    'Select the type of problem (R for Regression, C for Classification): ', ['R', 'C'])

if problem_type == 'R':
    dataset = get_input(
        'Select the dataset (S for Students, N for News): ', ['S', 'N'])
    testsize = get_float('Type the test size (0-1): ')
    if dataset == 'S':
        run_student(testsize)
    elif dataset == 'N':
        run_news(testsize)

elif problem_type == 'C':
    dataset = get_input(
        'Select the dataset (H for Heart Disease, F for Fraud Detection): ', ['H', 'F'])
    testsize = get_float('Type the test size (0-1): ')
    if dataset == 'H':
        run_heart(testsize=testsize)
    elif dataset == 'F':
        run_fraud(testsize=testsize)

print("Thank you for using the Machine Learning Model Runner!")
