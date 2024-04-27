from math import sqrt
ALPHA = 0.01
ITERATIONS = 20000


class Model:
    intercept = 0.0
    slope = 0.0

    _min_mileage: float | None = None
    _max_mileage: float | None = None
    _min_price:   float | None = None
    _max_price:   float | None = None

    def __init__(
            self,
            price: list[float] | None = None,
            mileage: list[float] | None = None,
            intercept: float | None = None,
            slope: float | None = None) -> None:
        self.price = price
        self.mileage = mileage
        if intercept is not None:
            self.intercept = intercept
        if slope is not None:
            self.slope = slope

    def predict(self, mileage: float) -> float:
        return self.intercept + self.slope * mileage

    def _compute_gradient(self):
        gradient_intercept = 0
        gradient_slope = 0
        mse = 0
        m = len(self.mileage)
        for mileage, price in zip(self.mileage, self.price):
            estimation = self.predict(mileage)
            gradient_intercept += estimation - price
            gradient_slope += (estimation - price) * mileage
        return gradient_intercept / m, gradient_slope / m, mse

    def _update_parameters(self, ALPHA) -> None:
        gradient_intercept, gradient_slope, mse = self._compute_gradient()
        self.intercept -= ALPHA * gradient_intercept
        self.slope -= ALPHA * gradient_slope

    def train(self, learning_rate=ALPHA, iterations=ITERATIONS) -> None:
        self._normalize_data()
        for _ in range(iterations):
            self._update_parameters(learning_rate)
        self._denormalize_data()
        self._denormalize_parameters()

    def compute_cost(self) -> float:
        total_squared_error = 0
        for i in range(len(self.mileage)):
            estimated_price = self.predict(self.mileage[i])
            error = estimated_price - self.price[i]
            total_squared_error += error ** 2
        mse = 1 / (2 * len(self.mileage)) * total_squared_error
        root_mean_squared_error = sqrt(mse)
        return root_mean_squared_error

    def _normalize_data(self):
        self._min_mileage = min(self.mileage)
        self._max_mileage = max(self.mileage)
        self._min_price, self._max_price = min(self.price), max(self.price)
        for i in range(len(self.mileage)):
            self.mileage[i] = (self.mileage[i] - self._min_mileage) /\
                (self._max_mileage - self._min_mileage)

        for i in range(len(self.price)):
            self.price[i] = (self.price[i] - self._min_price)\
                / (self._max_price - self._min_price)

    def _denormalize_data(self):
        for i in range(len(self.mileage)):
            self.mileage[i] = self.mileage[i] *\
                (self._max_mileage - self._min_mileage) + self._min_mileage

        for i in range(len(self.price)):
            self.price[i] = self.price[i] *\
                (self._max_price - self._min_price) + self._min_price

    def _denormalize_parameters(self):
        self.slope = self.slope * (self._max_price - self._min_price) /\
            (self._max_mileage - self._min_mileage)
        self.intercept = self.intercept *\
            (self._max_price - self._min_price) + self._min_price - self.slope\
            * self._min_mileage

    def hyperparameter_tuning(self, learning_rates, iterations):
        best_rmse = float('inf')
        best_learning_rate = None
        best_iterations = None

        for lr in learning_rates:
            for it in iterations:
                model = Model(price=self.price, mileage=self.mileage)
                model.train(learning_rate=lr, iterations=it)
                rmse = model.compute_cost()
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_learning_rate = lr
                    best_iterations = it

        return best_learning_rate, best_iterations, best_rmse
