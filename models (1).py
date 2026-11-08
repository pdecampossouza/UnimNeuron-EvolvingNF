import numpy as np

from numpy.linalg import pinv
from scipy.stats import norm
from pyswarm import pso
import tensorflow as tf
from sklearn.discriminant_analysis import StandardScaler

try:
    from tf.keras.models import Sequential  # type: ignore
    from tensorflow.keras.layers import Dense  # type: ignore
    from tensorflow.keras.metrics import AUC, Precision, Recall  # type: ignore

    _TF_AVAILABLE = True
except Exception:
    _TF_AVAILABLE = False
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LassoCV, lasso_path
from sklearn.utils import resample

from experiments.calculate import calculate_consistency_matrix
from models.operators import andneuron, orneuron, unineuron
from experiments.plots import visualize_membership_functions


class UniNormOperator:
    def __init__(self, g):
        self.g = g
        self.operation_type = None

    def t_norm(self, x, y):
        # Implementação do produto como t-norma
        return x * y

    def s_norm(self, x, y):
        # Implementação da soma probabilística como s-norma
        return x + y - x * y

    def __call__(self, x, y):
        if x < self.g and y < self.g:
            self.operation_type = "AND"
            return self.g * self.t_norm(x / self.g, y / self.g)
        elif x > self.g and y > self.g:
            self.operation_type = "OR"
            return self.g + (1 - self.g) * self.s_norm(
                (x - self.g) / (1 - self.g), (y - self.g) / (1 - self.g)
            )
        else:
            if self.g > 0.5:
                self.operation_type = "AND"
                return min(x, y)
            else:
                self.operation_type = "OR"
                return max(x, y)


class FNNModel:
    def __init__(
        self,
        num_mfs,
        neuron_type="andneuron",
        activation="linear",
        optimizer="moore-penrose",
        visualizeMF=False,
        pruning="none",
        rng_seed=None,
    ):
        """
        Initialize a Fuzzy Neural Network (FNN) model with the specified configuration.

        Parameters:
        - num_mfs (int): Number of membership functions for each input dimension.
        - neuron_type (str): Type of neuron to use in the FNN model. Default is "andneuron".
        - activation (str): Activation function to use in the FNN model. Default is "linear".
        - optimizer (str): Optimizer algorithm to use for training the FNN model. Default is "moore-penrose".
        - visualizeMF (bool): Whether to visualize membership functions during training. Default is False.
        - rng_seed (int or None): Seed for random number generation. If None, a default RNG is used. Default is None.

        Returns:
        None
        """

        self.model = None
        self.num_mfs = num_mfs
        self.neuron_type = neuron_type
        self.activation = activation
        self.pruning = pruning
        self.optimizer = optimizer
        self.visualizeMF = visualizeMF

        if rng_seed is None:
            self.rng_seed = np.random.default_rng(0)
        else:
            self.rng_seed = rng_seed

        self.mf_params = []
        self.neuron_weights = None
        self.total_fuzzy_neurons = None
        self.rules_dictionary = []  # Stores fuzzy rules
        self.axioms = []  # Stores axioms generated from fuzzy rules

    def calculate_mutual_information(x, y):
        mi_scores = np.array([mutual_info_score(x[:, i], y) for i in range(x.shape[1])])
        return mi_scores / mi_scores.sum()  # Normalize MI scores

    def fuzzification_layer(self, x):
        """
        Performs the fuzzification layer operation for the input data.

        Parameters:
        - x (numpy.ndarray): Input data of shape (num_samples, num_features).

        Returns:
        - fuzzy_outputs (numpy.ndarray): Fuzzy outputs after fuzzification, with shape (num_samples, total_fuzzy_neurons).
        Each row represents the fuzzy output for a sample, and each column represents the degree of membership to a fuzzy set.
        """

        num_samples, num_features = x.shape
        if self.total_fuzzy_neurons is None:
            self.total_fuzzy_neurons = self.num_mfs**num_features

        fuzzy_outputs = np.zeros((num_samples, self.total_fuzzy_neurons))
        # If the parameters have not been created yet
        if len(self.mf_params) == 0:
            self.total_fuzzy_neurons = self.num_mfs**num_features
            self.mf_params = [
                {"centers": [], "sigmas": []} for _ in range(num_features)
            ]
            for feature_index in range(num_features):
                feature_min, feature_max = np.min(x[:, feature_index]), np.max(
                    x[:, feature_index]
                )
                centers = np.linspace(feature_min, feature_max, self.num_mfs)
                sigmas = np.full_like(
                    centers,
                    (feature_max - feature_min) / (self.num_mfs * np.sqrt(2 * np.pi)),
                )

                self.mf_params[feature_index] = {"centers": centers, "sigmas": sigmas}
                # self.rules_dictionary.append({"centers": centers, "sigmas": sigmas})

                if self.visualizeMF:
                    visualize_membership_functions(feature_index, centers, sigmas)

        # Get centers and sigmas, and compute fuzzy outputs
        for feature_index in range(num_features):
            for mf_index in range(self.num_mfs):
                sigma = self.mf_params[feature_index]["sigmas"][mf_index]
                center = self.mf_params[feature_index]["centers"][mf_index]
                gaussian_output = norm.pdf(x[:, feature_index], center, sigma)
                fuzzy_outputs[:, feature_index * self.num_mfs + mf_index] = (
                    gaussian_output
                )
        if self.neuron_weights is None:
            self.neuron_weights = self.rng_seed.random(
                (self.total_fuzzy_neurons, num_features)
            )  # Generates random weights
        self.rules_dictionary = self.generate_rules_dictionary()
        return fuzzy_outputs  # Return the fuzzy outputs

    def generate_rules_dictionary(self):
        """
        Generates a dictionary of rules based on combinations of MFs for all characteristics.
        """
        rules_dictionary = []
        num_features = len(self.mf_params)
        num_mfs_per_feature = self.num_mfs
        total_rules = num_mfs_per_feature**num_features

        for rule_index in range(total_rules):
            mf_combination = np.unravel_index(
                rule_index, [num_mfs_per_feature] * num_features
            )
            rule_params = {"centers": [], "sigmas": []}
            for feature_index, mf_index in enumerate(mf_combination):
                rule_params["centers"].append(
                    self.mf_params[feature_index]["centers"][mf_index]
                )
                rule_params["sigmas"].append(
                    self.mf_params[feature_index]["sigmas"][mf_index]
                )
            rules_dictionary.append(rule_params)

        return rules_dictionary

    def adjust_weights_by_density(self, x):
        num_samples, num_features = x.shape
        densities = np.zeros((self.total_fuzzy_neurons, num_features))

        # Normalizando os dados
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        # Calcule a densidade para cada neurônio gaussiano
        for feature_index in range(num_features):
            feature_data = x_scaled[:, feature_index].reshape(-1, 1)
            for mf_index in range(self.num_mfs):
                center = self.mf_params[feature_index]["centers"][mf_index]
                sigma = self.mf_params[feature_index]["sigmas"][mf_index]

                # Ajuste o bandwidth baseado no sigma
                bandwidth = max(sigma, 0.3)  # Evita bandwidth muito pequeno

                # KDE para calcular a densidade no centro
                kde = KernelDensity(bandwidth=bandwidth)
                kde.fit(feature_data)

                # Preparando uma amostra fictícia para transformação
                sample = np.zeros((1, num_features))  # cria uma amostra com zeros
                sample[0, feature_index] = (
                    center  # define o valor do centro na característica relevante
                )

                # Converta o centro para a escala transformada
                center_scaled = scaler.transform(sample)[0][feature_index]
                density = np.exp(kde.score_samples([[center_scaled]]))
                densities[feature_index * self.num_mfs + mf_index, feature_index] = (
                    density
                )

        # Normalizar densities usando Min-Max Scaling para evitar colapsos a zero
        min_density = np.min(densities)
        max_density = np.max(densities)
        normalized_densities = (
            (densities - min_density) / (max_density - min_density)
            if max_density > min_density
            else np.zeros_like(densities)
        )

        # Garantindo que todos os valores sejam não-negativos
        normalized_densities = np.clip(normalized_densities, 0, 1)

        # Update weights to reflect relative importance
        self.neuron_weights *= (
            normalized_densities + 0.5
        )  # Ensures no weight is zero; adjust constant as needed
        # Limits weight values between 0 and 1
        self.neuron_weights = np.clip(self.neuron_weights, 0, 1)
        return self.neuron_weights

    def logic_neurons_layer(self, fuzzy_outputs):
        """
        Calculates the logical output of the fuzzy neural network based on the fuzzy outputs and neuron weights.

        Parameters:
        - fuzzy_outputs (numpy.ndarray): Fuzzy outputs after fuzzification, with shape (num_samples, total_fuzzy_neurons).
          Each row represents the fuzzy output for a sample, and each column represents the degree of membership to a fuzzy set.
        Returns:
        - logic_output (numpy.ndarray): Logical output of the fuzzy neural network, with shape (num_samples, total_fuzzy_neurons).
          Each row represents the logical output for a sample, and each column represents the output of a logical neuron
          corresponding to a specific combination of membership functions for input features.
        """

        num_samples = fuzzy_outputs.shape[0]
        num_features = len(self.mf_params)
        num_mfs_per_feature = self.num_mfs
        self.total_fuzzy_neurons = num_mfs_per_feature**num_features

        # Initializes the logical output vector `z`
        z = np.zeros((num_samples, self.total_fuzzy_neurons))

        # Iterates over all possible combinations of MFs for all features
        for sample_index in range(num_samples):
            for neuron_index in range(self.total_fuzzy_neurons):
                # Obtains the specific combination of MFs for the current neuron index
                mf_combination = np.unravel_index(
                    neuron_index, [num_mfs_per_feature] * num_features
                )

                # Calculates the logical neuron output for the MF combination
                logic_neuron_input = [
                    fuzzy_outputs[
                        sample_index, feature_index * num_mfs_per_feature + mf_index
                    ]
                    for feature_index, mf_index in enumerate(mf_combination)
                ]
                weights = self.neuron_weights[neuron_index]

                if self.neuron_type == "andneuron":
                    z[sample_index, neuron_index] = andneuron(
                        logic_neuron_input, weights
                    )
                elif self.neuron_type == "orneuron":
                    z[sample_index, neuron_index] = orneuron(
                        logic_neuron_input, weights
                    )
                elif self.neuron_type == "unineuron":
                    g_value = np.random.rand()
                    uni_op = UniNormOperator(g_value)
                    z[sample_index, neuron_index] = self.unineuron(
                        logic_neuron_input, weights, g_value, neuron_index, uni_op
                    )

        logic_outputs = z
        # Se os índices ativos dos neurônios estão definidos, usa apenas esses para a saída
        if (
            hasattr(self, "active_neurons_indices")
            and len(self.active_neurons_indices) > 0
        ):
            logic_outputs[:, self.active_neurons_indices]
        return logic_outputs

    def train_model(self, x_train, y_train):
        """
        Trains the fuzzy neural network model using the provided training data.

        Parameters:
        - x_train (numpy.ndarray): Input features of the training data.
        - y_train (numpy.ndarray): Target labels of the training data.

        Returns:
        None
        """

        # Adjust the trainModel method to capture the return value from fuzzificationLayer
        fuzzy_outputs = self.fuzzification_layer(x_train)  # Capture the fuzzy outputs
        logic_outputs = self.logic_neurons_layer(
            fuzzy_outputs
        )  # Pass the fuzzy outputs to the logic neurons layer

        if self.optimizer == "moore-penrose":
            # Use Moore-Penrose pseudo-inverse for training
            self.V = np.dot(pinv(logic_outputs), y_train)
            self.VR = self.V.flatten()
        elif self.optimizer == "adam":
            # Use a neural network for training
            self.neural_network_layer(logic_outputs, y_train)
            last_layer_weights = self.model.layers[-1].get_weights()[0]
            # Flatten the weights to make them consistent with V format and store in VR
            self.VR = np.array((last_layer_weights.flatten()))
            self.V = last_layer_weights.flatten()
        elif self.optimizer == "pso":
            # Utilize o PSO para otimizar os pesos
            # Y_train_pso = np.where(y_train == -1, 0, y_train)
            # optimized_weights = self.psoOptimize(
            #     logic_outputs, Y_train_pso, num_particles=200, max_iter=5000
            # )
            # self.V = optimized_weights
            # self.VR = np.array(self.V.flatten())  # Achatando V para consistência

            # PSO parameters
            dim = (
                logic_outputs.shape[1] * y_train.shape[1]
            )  # Dimension of the weights to optimize
            lb = -1  # Lower bound of the weights
            ub = 1  # Upper bound of the weights
            num_particles = 1000
            max_iter = 100
            optimized_weights = self.psoOptimize(
                logic_outputs, y_train, num_particles, max_iter
            )

            # Update the model's output layer weights with the optimized solution
            self.V = optimized_weights.reshape(logic_outputs.shape[1], -1)
            self.VR = np.array(self.V.flatten())  # Flatten V for consistency

        print(f"Number of rules before pruning: {self.total_fuzzy_neurons}")
        # Prune neurons based on consistency after defining V
        if self.pruning == "consistency":
            keep_mask = self.prune_inconsistent_neurons(self.V, threshold=0.1)
            self.update_model_structure(logic_outputs, keep_mask)
            print(f"Number of rules after pruning: {np.sum(keep_mask)}")
        elif self.pruning == "bolasso":
            y_convert = np.where(y_train == -1, 0, y_train)
            y_train_converted = np.ravel(y_convert)
            keep_mask = self.bolasso(
                logic_outputs,
                y_train_converted,
                n_bootstraps=32,
                consensus_threshold=0.1,
            )
            self.update_model_structure(logic_outputs, keep_mask)
            print(f"Number of rules after pruning: {np.sum(keep_mask)}")
            print("Indices of active rules:", np.where(keep_mask)[0])
        else:
            print("Unsupported pruning method. Proceding with all the neurons")
            self.Oldtotal_fuzzy_neurons = self.total_fuzzy_neurons
            self.OldV = self.V

    def update_model_structure(self, logic_outputs, keep_mask):
        """
        Updates the model's structure based on the pruning mask.

        Parameters:
        - logic_outputs (numpy.ndarray): The logical outputs from the neurons.
        - keep_mask (numpy.ndarray): A boolean array indicating which neurons are kept.

        Returns:
        None
        """
        # Update logical outputs and weights to only include kept neurons
        self.logic_outputs = logic_outputs[:, keep_mask]
        self.active_neurons_indices = np.where(keep_mask)[
            0
        ]  # Store active indices for later use
        self.OldV = self.V
        self.V = self.V[keep_mask]
        self.Oldrules_dictionary = self.rules_dictionary
        self.rules_dictionary = [
            rule for k, rule in zip(keep_mask, self.rules_dictionary) if k
        ]
        self.Oldtotal_fuzzy_neurons = self.total_fuzzy_neurons
        self.total_fuzzy_neurons = len(
            self.V
        )  # Update the total of neurons after pruning

    def prune_inconsistent_neurons(self, V, threshold):
        """
        Prunes neurons based on consistency analysis of the rule consequents.

        Parameters:
        - V (numpy.ndarray): Neuron weights array.
        - threshold (float): Consistency threshold for pruning.

        Returns:
        - numpy.ndarray: Boolean array indicating which neurons are kept.
        """
        consistency_matrix = calculate_consistency_matrix(V)
        avg_consistency = np.mean(consistency_matrix, axis=0)
        return avg_consistency >= threshold

    def bolasso(self, X, y, n_bootstraps, consensus_threshold):
        n_samples, n_features = X.shape
        selected_variables = np.zeros((n_bootstraps, n_features), dtype=bool)

        for i in range(n_bootstraps):
            X_resampled, y_resampled = resample(X, y)
            lasso = LassoCV(cv=5).fit(X_resampled, y_resampled)

            _, coef_path, _ = lasso_path(X_resampled, y_resampled, alphas=lasso.alphas_)
            # Checking if any coefficient is non-zero along the regularization path
            selected_variables[i] = np.any(coef_path != 0, axis=1)

        variable_frequency = np.mean(selected_variables, axis=0)
        consensus_variables = variable_frequency >= consensus_threshold
        X_consensus = X[:, consensus_variables]
        final_model = LassoCV(cv=5).fit(X_consensus, y)

        return consensus_variables

    def psoOptimize(self, X, Y, num_particles, max_iter):
        # Define the limits of the search space
        dim = X.shape[1] * (Y.shape[1] if len(Y.shape) > 1 else 1)
        lb = -1 * np.ones(dim)
        ub = 1 * np.ones(dim)

        def fitness_function(weights):
            return self.fitness(weights, X, Y)

        # Perform PSO optimization using pyswarm
        optimized_weights, _ = pso(
            fitness_function,
            lb,
            ub,
            swarmsize=num_particles,
            maxiter=max_iter,
            debug=True,
        )

        return optimized_weights

    def fitness(self, weights, X, Y):
        # Reshape weights to match the model's output layer structure
        num_outputs = Y.shape[1] if len(Y.shape) > 1 else 1
        weights = weights.reshape((X.shape[1], num_outputs))

        # Temporarily set the model's output layer weights to the candidate solution
        self.V = weights

        # Generate predictions for the given inputs
        y_pred = np.dot(X, self.V)  # Garanta que `self.logic_output` esteja correto
        Y_pred_binary = np.sign(
            y_pred
        )  # Ajuste se seus dados de saída forem diferentes

        # Calculate accuracy
        accuracy = np.mean(Y_pred_binary == Y)

        # Return the inverse of accuracy because PSO minimizes the fitness function
        # and we want to maximize accuracy
        return 1 - accuracy

    def neural_network_layer(self, x, y):
        self.model = Sequential(
            [
                Dense(
                    1,
                    input_dim=x.shape[1],
                    activation="linear",
                    kernel_initializer="random_uniform",
                    bias_initializer="zeros",
                )
            ]
        )
        # Adds AUC, precision, and recall to the metrics
        self.model.compile(
            optimizer="sgd",
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                AUC(name="auc"),
                Precision(name="precision"),
                Recall(name="recall"),
            ],
        )
        self.model.fit(x, y, epochs=200, batch_size=1)

    def evaluate_model(self, x_test, y_test):
        """
        Evaluate the trained FNN model using test data.

        Parameters:
            x_test (numpy.ndarray): Features of the test data.
            y_test (numpy.ndarray): Labels of the test data.

        Returns:
            evaluation_metrics (dict): Dictionary containing evaluation metrics including accuracy, specificity,
            precision, recall, and F-score.
        """
        # Evaluate the trained FNN model
        evaluation_metrics = {}
        fuzzy_outputs_test = self.fuzzification_layer(x_test)
        logic_outputs_test = self.logic_neurons_layer(fuzzy_outputs_test)
        # Select only the columns corresponding to active neurons after pruning
        if hasattr(self, "active_neurons_indices"):
            logic_outputs_test = logic_outputs_test[:, self.active_neurons_indices]
        if self.optimizer in ["moore-penrose", "pso"]:
            Y_pred = np.dot(logic_outputs_test, self.V)
            Y_pred_binary = np.sign(Y_pred)
            accuracy = np.mean((Y_pred_binary) == y_test)
            tp = np.sum((Y_pred_binary == 1) & (y_test == 1))
            tn = np.sum((Y_pred_binary == -1) & (y_test == -1))
            fp = np.sum((Y_pred_binary == 1) & (y_test == -1))
            fn = np.sum((Y_pred_binary == -1) & (y_test == 1))

            epsilon = 1e-8

            # Ajuste das métricas para evitar NaN
            recall = tp / (tp + fn + epsilon)
            specificity = tn / (tn + fp + epsilon)
            precision = tp / (tp + fp + epsilon)

            # Para evitar NaN no cálculo do f_score, também aplicamos o epsilon na fórmula
            f_score = 2 * (precision * recall) / (precision + recall + epsilon)
            evaluation_metrics["accuracy"] = round(float(accuracy), 3)
            evaluation_metrics["specificity"] = round(specificity, 3)
            evaluation_metrics["precision"] = round(precision, 3)
            evaluation_metrics["recall"] = round(recall, 3)
            evaluation_metrics["fscore"] = round(f_score, 3)
            # Prints the metrics
            print(f"\nAccuracy: {accuracy * 100:.2f}%")
            print(f"Specificity: {specificity}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F-Score: {f_score}")
        else:
            scores = self.model.evaluate(logic_outputs_test, y_test)
            y_pred = self.model.predict(self.logic_output)
            Y_pred_binary = np.sign(y_pred)
            print(f"\nAccuracy: {scores[1] * 100:.2f}%")

            evaluation_metrics["accuracy"] = scores[1]

        return Y_pred_binary, evaluation_metrics

    def generate_fuzzy_rules(self):
        """
        Generate fuzzy rules based on the weights of the neurons in the logic layer.

        Returns:
        list: A list of fuzzy rules, where each rule is represented as a string.
        """
        # Preparation of rule strings
        rules = []

        # For each neuron in the logic layer
        for neuron_index in range(self.Oldtotal_fuzzy_neurons):
            # Obtains the specific combination of MFs for the current neuron index
            mf_combination = np.unravel_index(
                neuron_index, [self.num_mfs] * len(self.mf_params)
            )
            rule = "IF "

            # For each feature, add the corresponding MF to the rule
            for feature_index, mf_index in enumerate(mf_combination):
                weight = self.neuron_weights[
                    neuron_index, feature_index
                ]  # Access the specific weight
                rule += (
                    f"x{feature_index+1} is MF{mf_index+1} with impact {weight:.2f} "
                )
                if feature_index < len(mf_combination) - 1:
                    if self.neuron_type == "unineuron":
                        operation_type = self.neuron_details[neuron_index][
                            "operation_type"
                        ]
                        rule += "AND " if operation_type == "AND" else "OR "
                    else:
                        rule += "AND " if self.neuron_type == "andneuron" else "OR "

            # Adds the rule's result (output)
            rule += f"THEN output is {self.OldV[neuron_index]}"

            # Adds the complete rule to the list of rules
            rules.append(rule)

        return rules

    # TO ME, THE FOLLOWING FUNCTION SEEMS THE SAME OF PREVIOUS ONE...
    # IF YOU HAVE TO USE IT DECOMMENT IT AND ADD DOCUMENTATION

    # def generate_fuzzy_rules_with_impact(self):
    #     rules_with_impact = []
    #
    #     # For each neuron in the logic layer
    #     for neuron_index in range(self.total_fuzzy_neurons):
    #         # Obtains the specific combination of MFs for the current neuron index
    #         mf_combination = np.unravel_index(
    #             neuron_index, [self.num_mfs] * len(self.mf_params)
    #         )
    #         rule_with_impact = "IF "
    #
    #         # For each feature, add the corresponding MF to the rule with its weight (impact)
    #         for feature_index, mf_index in enumerate(mf_combination):
    #             weight = self.neuron_weights[
    #                 neuron_index, feature_index
    #             ]  # Access the specific weight
    #             rule_with_impact += (
    #                 f"x{feature_index+1} is MF{mf_index+1} with impact {weight:.2f} "
    #             )
    #             if feature_index < len(mf_combination) - 1:
    #                 rule_with_impact += (
    #                     "AND " if self.neuron_type == "andneuron" else "OR "
    #                 )
    #
    #             # Adds the rule's result (output)
    #         rule_with_impact += f"THEN output is {self.V[neuron_index]}"
    #
    #     # Adds the complete rule to the list of rules
    #     rules_with_impact.append(rule_with_impact)
    #
    #     return rules_with_impact

    # IF YOU HAVE TO USE IT DECOMMENT IT AND ADD DOCUMENTATION, STILL HAVE SOME DOUBTS ON IT

    # def visualize_3d_logic_neuron_output(self, X1_range, X2_range, resolution=100):
    #     """
    #     Visualize 3D projection of a logical neuron output.
    #
    #     Parameters:
    #     - X1_range: Tuple (min, max) for the first input feature range.
    #     - X2_range: Tuple (min, max) for the second input feature range.
    #     - resolution: Number of points per dimension in the grid.
    #     """
    #     # Generate a grid of points within the specified ranges
    #     x1 = np.linspace(X1_range[0], X1_range[1], resolution)
    #     x2 = np.linspace(X2_range[0], X2_range[1], resolution)
    #     X1, X2 = np.meshgrid(x1, x2)
    #
    #     # Flatten the grid to pass through the fuzzification layer
    #     grid_flat = np.c_[X1.ravel(), X2.ravel()]
    #
    #     # Fuzzify the grid points
    #     fuzzy_outputs = self.fuzzification_layer(grid_flat)
    #
    #     # Compute the logical outputs
    #     self.logic_neurons_layer()
    #
    #     # Assuming self.logic_output now contains the outputs for the grid, reshape it back to the grid shape
    #     Z = self.logic_output.reshape(X1.shape)
    #
    #     # Create a 3D plot
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection="3d")
    #
    #     # Plot the surface
    #     surf = ax.plot_surface(X1, X2, Z, cmap="viridis", edgecolor="none")
    #     fig.colorbar(surf, shrink=0.5, aspect=5)  # Add a color bar to indicate the values
    #
    #     ax.set_xlabel("Input Feature 1")
    #     ax.set_ylabel("Input Feature 2")
    #     ax.set_zlabel("Logical Neuron Output")
    #     ax.set_title("3D Projection of Logical Neuron Output")
    #
    #     plt.show()

    def generate_fuzzy_axioms(self):
        self.axioms.clear()

        # --- Normalize V ---
        V = np.asarray(self.V, dtype=float)
        if V.ndim > 1:
            V = V[:, 0]

        for neuron_index, v in enumerate(V):

            mf_combination = np.unravel_index(
                neuron_index, [self.num_mfs] * len(self.mf_params)
            )

            rule_parts = []

            for feature_index, mf_index in enumerate(mf_combination):
                center = self.mf_params[feature_index]["centers"][mf_index]
                sigma = self.mf_params[feature_index]["sigmas"][mf_index]

                interval_start = round(center - sigma, 2)
                interval_end = round(center + sigma, 2)

                rule_parts.append(
                    f"x{feature_index+1} is around "
                    f"[{interval_start} - {interval_end}] "
                    f"with sigma {sigma:.2f}"
                )

            conjunction = " AND " if self.neuron_type == "andneuron" else " OR "
            if self.neuron_type == "unineuron":
                conjunction = self.neuron_details[neuron_index]["operation_type"]

            rule_body = conjunction.join(rule_parts)

            axiom = f"IF {rule_body}, THEN output is [{round(float(v), 2)}]."
            self.axioms.append(axiom)

        return self.axioms
