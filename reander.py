import matplotlib.pyplot as plt
from linear_regression.model import Model
from linear_regression.parseing import parse_data, parse_model


def show_cost(mileage, price, theta0, theta1):

    # Créer le tracé
    fig, (ax1, ax2) = plt.subplots(
        1, 2, constrained_layout=True,
        figsize=(16, 8))
    # Tracer l'évolution du coût pour les 100 premières itérations
    ax1.plot(mileage, price, 'blue', marker='o', linestyle='None', label="Car")

    ax1.set_title("data")
    ax1.set_ylabel('Price')
    ax1.set_xlabel('Mileage')

    ax2.plot(mileage, price, 'blue', marker='o', linestyle='None', label="Car")
    x_min = min(mileage)
    x_max = max(mileage)
    y_min = theta0 + theta1 * x_min
    y_max = theta0 + theta1 * x_max
    ax2.plot(
        [x_min, x_max], [y_min, y_max],
        label="Linear regression", color='red', linewidth=2
    )
    ax2.set_title("Linear regression")
    ax2.set_ylabel('Price')
    ax2.set_xlabel('Mileage')

    ax1.legend()
    ax2.legend()

    fig.savefig('data/plot_rain.png')
    # Afficher le tracé
    plt.show()


def main():
    mileage, price = parse_data('data/data.csv')
    if mileage and price is None:
        return
    theta0, theta1 = parse_model('data/model.csv')
    model = Model(price=price, mileage=mileage)
    model.train()
    show_cost(mileage, price, theta0, theta1)


if __name__ == '__main__':
    main()
