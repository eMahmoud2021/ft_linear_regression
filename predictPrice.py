import sys
from linear_regression.model import Model
from linear_regression.parseing import parse_model


def getMileage():
    try:
        mileage = input('Enter the mileage of the car: ')

        mileage = float(mileage)
        if mileage < 0:
            raise ValueError('mileage cannot be negative')

        return mileage

    except KeyboardInterrupt:
        return None
    except Exception as e:
        print(f'Invalid mileage: {e}')
        return None


def main():
    input_mileage = getMileage()
    print(f'You entered: {input_mileage}')

    try:
        intercept, slope = parse_model('data/model.csv')
    except (FileNotFoundError, Exception):
        print(f"The estimated price for a car with {input_mileage} km is\
            {0} €. Your model hasn't been trained yet.")
        sys.exit(1)  # Exit with error code 1
    if intercept is None or slope is None:
        print(f'intercept not valide: {intercept}  slope not valide')
        
        sys.exit(1)  # Exit with error code 1

    model = Model(intercept=intercept, slope=slope)
    price = model.predict(input_mileage)

    if price < 0:
        print("Don't buy this shit:", format(price, '.2f'))
    else:
        print(f'The estimated price for a car with {input_mileage} km is\
            {price:.2f} €.')


if __name__ == '__main__':
    main()
