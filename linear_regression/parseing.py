import csv


def getfile(file_path: str, mode: str):
    try:
        with open(file_path, mode) as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            data = list(reader)
            print(f"File opened successfully: {file_path} with mode: {mode}")
            return data
    except Exception as e:
        print(f"Error file: {e}")
    return None


def save_model(file_path: str, intercept: float, slope: float) -> None:
    """
        Save the model to the file.
    """
    try:
        with open(file_path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['intercept', 'slope'])
            writer.writerow([intercept, slope])
        print(f'Saved model with intercept: {intercept},\
            slope: {slope} to `{file_path}`.')
    except (FileNotFoundError, IndexError, ValueError) as e:
        print.error(f'Could not save model: {e}', exc_info=False)


def parse_model(file_path: str):
    """
    Parse the model from the file.
    """
    try:
        reader = getfile(file_path, 'r')
        if reader:
            intercept, slope = float(reader[0][0]), float(reader[0][1])
            if isinstance(intercept, float) and isinstance(slope, float):
                print(f'Parsed model with intercept: {intercept},\
                    slope: {slope}')
                return intercept, slope
            else:
                print(f'Invalid values in model file: `{file_path}`')
                return None, None
        else:
            print(f'Could not parse model from file: `{file_path}`')
            return None, None
    except (ValueError, IndexError):
        print(f'Invalid values in model file: `{file_path}`')
        return None, None
    except Exception:
        print(f'Could not parse model from file: `{file_path}`')
        return None, None


def parse_data(file_path: str):
    mileag = []
    price = []
    try:
        reader = getfile(file_path, 'r')
        for row in reader:
            if row is not None:
                mileag.append(float(row[0]))
                price.append(float(row[1]))
    except Exception as e:
        print(f"Error while parsing data: {e}")
    return mileag, price


