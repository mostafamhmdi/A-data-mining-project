
from .student_performance import *

problem_type = input(
    'Select the type of problem: (R for Regression and L for ranking)')

if problem_type == 'R':
    dataset = input(
        'Select the dataset: (S for students dataset and N for news dataset)')
    if dataset == 'S':
        run_student()
    elif dataset == 'N':
