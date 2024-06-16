
from studentperformance import *
# from newsshares import *

problem_type = input(
    'Select the type of problem: (R for Regression and C for classification)   ')

if problem_type == 'R':
    dataset = input(
        'Select the dataset: (S for students dataset and N for news dataset)   ')
    testsize = int(input(
        'Type the test size(0-100)'))
    if dataset == 'S':
        run_student(testsize)
#     elif dataset == 'N':
#         run_news(testsize)


# if problem_type == 'C':
#     dataset = input(
#         'Select the dataset: (H for heart disease dataset and F for fraud detection dataset)')
#     testsize = int(input(
#         'Type the test size(0-100)'))
#     if dataset == 'H':
#         run_student()
#     elif dataset == 'F':
#         run_news()
