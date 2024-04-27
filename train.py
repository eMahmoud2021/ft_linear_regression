from linear_regression.model import Model
from linear_regression.parseing import save_model, parse_data


def main():
    mileage, price = parse_data('data/data.csv')
    if mileage and price is None:
        return
    model = Model(price=price, mileage=mileage)
    model.train()
    save_model('data/model.csv', model.intercept, model.slope)
    cost = model.compute_cost()
    print(f'The Root Mean Squared Error of the model is {cost:.2f}.')

    learning_rates_to_try = [0.01, 0.1, 0.5, 0.0001]
    iterations_to_try = [1000, 200000, 10000, 210000]

    # Call the hyperparameter tuning method
    best_lr, best_iterations, best_rmse = model.hyperparameter_tuning(
        learning_rates_to_try, iterations_to_try)

    # Print the results
    print("Best Learning Rate:", best_lr)
    print("Best Iterations:", best_iterations)
    print("Best RMSE:", best_rmse)

    
if __name__ == '__main__':
    main()
