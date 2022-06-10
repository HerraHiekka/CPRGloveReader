import numpy as np

def is_float(element) -> bool:
    element = np.array(element)
    try:
        element.astype('float')
        return True
    except ValueError:
        return False

if __name__ == '__main__':
    print(is_float('15'))
    print(is_float(['15']))
    print(is_float(['15','15.0','12']))
    print(is_float([]))
    print(is_float(['15.0']))
    print(is_float(['�19']))
    print(is_float(['a19']))
