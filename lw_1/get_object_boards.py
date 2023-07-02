import math


def get_object_outlines(proj, tol):

    init_val = proj[0]
    low_found = False
    high_found = False
    for i, val in enumerate(proj):

        if math.fabs(init_val - val) >= tol and not low_found:
            low_found = True
            high_found = False
            print(f'Low outline coordinates: {i}')
        if math.fabs(init_val - val) < tol and low_found and not high_found:
            high_found = True
            low_found = False
            print(f'High outline coordinates: {i - 1}')
