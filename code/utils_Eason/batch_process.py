import os


def list_tester(path: str):
    contents = os.listdir(path)
    result = []
    for name in contents:
        try:
            test_name, extension = name.rsplit('.', 1)
            if extension == 'edf':
                result.append(test_name)
        except ValueError:
            continue
    return result
