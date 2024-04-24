import tensorflow as tf
import numpy as np


class NumericalOptimizer:
    # optimizes for a given metric
    # argmax sum_m sum_z sum_i qs[z]_i log Pr[x_i^m | z] for each allowed probability
    # subj to Pr[x_i^m] >= sum_z Pr[x_i^m | z]

    def __init__(self,
                 qs,
                 observed_values,
                 observed_value_probabilities,
                 new_priors,
                 metric_model,
                 max_iterations,
                 learning_rate,
                 decay_rate,
                 identificator,
                 distributions_to_use,
                 constraint_penalty=100,
                 pseudocount=1e-7,
                 tolerance=0.1,
                 report_step=None):
        # input data
        if report_step is None:
            report_step = max_iterations // 5

        self.qs = qs
        self.observed_values = observed_values  # only for a given m
        self.observed_value_probabilities = observed_value_probabilities
        self.new_priors = new_priors
        self.metric_model = metric_model
        self.pseudocount = pseudocount

        # initial_variables -- probability func parameters
        self.n_z = qs.shape[1]
        self.distributions_to_use = distributions_to_use
        self.tolerance = tolerance

        def define_param_variables(z):
            # get an initial param estimation of the distribution
            distribution = distributions_to_use[z]
            params = metric_model.fill_params_distribution(distribution, observed_values)

            # transform
            trainable_variable = tf.Variable(initial_value=params,
                                             trainable=True,
                                             dtype=tf.float64,
                                             name=f'{z}:{distribution}',
                                             constraint=metric_model.get_validifying_function(distribution)
                                             )
            return trainable_variable

        self.variables = [define_param_variables(z) for z in range(self.n_z)]

        # optimizer settings
        self.iterations = max_iterations
        self.ident = identificator
        self.report_step = report_step
        self.penalty = constraint_penalty
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate,
            decay_steps=max_iterations,
            decay_rate=decay_rate)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        # overall minimum
        self.general_achieved_minimum_params = None

    def overflow_sum(self, probabilities):
        sum_probabilities = tf.reduce_sum(probabilities, axis=1)
        diff = sum_probabilities - self.observed_value_probabilities
        diff = tf.nn.relu(diff)  # the value being below is okay
        return tf.reduce_sum(diff)

    def func_to_optimize(self, variables):
        #  sum_z sum_i qs[z]_i log Pr[x_i^m | z] for specified distribution
        #  subj to Pr[x_i^m] >= sum_z Pr[x_i^m | z]
        m_params = [[self.distributions_to_use[z], params] for z, params in zip(range(self.n_z), variables)]
        probabilities = tf.stack(
            [self.metric_model.calculate_metric_probability(z,
                                                            m_params=m_params,
                                                            m_observed_values=self.observed_values) for z, params in
             zip(
                 range(self.n_z), variables
             )]
        )
        if tf.math.reduce_any(tf.math.is_nan(probabilities)):
            print("NAN IN PROBABILITIES!")
        probabilities = tf.transpose(probabilities) + self.pseudocount
        log_probabilities = tf.math.log(probabilities)
        if tf.math.reduce_any(tf.math.is_nan(log_probabilities)):
            print("NAN IN LOG-PROBABILITIES!")
        to_sum = self.qs * log_probabilities
        objective = - tf.reduce_sum(to_sum)  # to make minimizer from maximizer

        constraint = self.overflow_sum(probabilities) * self.penalty

        return objective + constraint

    def optimize(self):
        # observed_values, observed_value_probabilities,
        # are of shape (n, 1)

        last_seen_loss = np.inf
        general_achieved_minimum_objective = np.inf

        for i in range(self.iterations):
            with tf.GradientTape() as tape:
                # Record the operations for automatic differentiation
                flat_variables = tf.nest.flatten(self.variables)
                tape.watch(flat_variables)

                loss = self.func_to_optimize(self.variables)

            # store the achieved minimum
            if loss.numpy() < general_achieved_minimum_objective:
                general_achieved_minimum_objective = loss.numpy()
                self.general_achieved_minimum_params = [[tf.constant(var.numpy()) for var in sublist] for sublist in
                                                        self.variables]

            # Compute gradients
            gradients = tape.gradient(loss, flat_variables)
            self.optimizer.apply_gradients(zip(gradients, flat_variables))

            # Reporting
            if i % self.report_step == 0:
                if np.abs(last_seen_loss - loss.numpy()) < self.tolerance:
                    print(
                        f"{self.ident}:\tChange from the last report is lower than {self.tolerance}, loss: {loss.numpy()}, ending run")
                    break
                last_seen_loss = loss.numpy()
                print(f"{self.ident}:\tOptimization iteration {i}, Loss: {loss.numpy()}")
        return loss

    def get_optimized_parameters(self):
        return self.general_achieved_minimum_params


class WeightedNumericalOptimizer:
    # optimizes over all metrics
    # argmax sum_m sum_z sum_i qs[z]_i w(m, z) log Pr[x_i^m | z] for each allowed probability
    # subj to Pr[x_i^m] >= sum_z Pr[x_i^m | z]
    def __init__(self,
                 qs,
                 observed_values,
                 observed_value_probabilities,
                 new_priors,
                 metric_models,
                 max_iterations,
                 learning_rate,
                 decay_rate,
                 identificator,
                 distributions_to_use,
                 constraint_penalty=100,
                 pseudocount=1e-7,
                 tolerance=0.1,
                 report_step=None):
        # input data
        if report_step is None:
            report_step = max_iterations // 5

        # initial_variables -- probability func parameters
        self.n_z = qs.shape[1]
        self.n_m = observed_values.shape[1]
        self.distributions_to_use = distributions_to_use
        self.tolerance = tolerance

        self.qs = qs
        self.observed_values = observed_values  # for all m
        self.observed_value_probabilities = observed_value_probabilities
        self.new_priors = tf.reshape(new_priors, (1, 1, self.n_z))
        self.metric_models = metric_models  # list -- for all m
        self.pseudocount = pseudocount

        def define_param_variables(m, z):
            # get an initial param estimation of the distribution
            distribution = distributions_to_use[m][z]
            metric_model = metric_models[m]
            params = metric_model.fill_params_distribution(distribution, observed_values[:, m])

            # transform
            trainable_variable = tf.Variable(initial_value=params,
                                             trainable=True,
                                             dtype=tf.float64,
                                             name=f'z={z}:m={m}:{distribution}',
                                             constraint=metric_model.get_validifying_function(distribution)
                                             )
            return trainable_variable

        # 2d list (m,z)
        self.variables = [[define_param_variables(m, z) for z in range(self.n_z)] for m in range(self.n_m)]

        # optimizer settings
        self.iterations = max_iterations
        self.ident = identificator
        self.report_step = report_step
        self.penalty = constraint_penalty
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate,
            decay_steps=max_iterations,
            decay_rate=decay_rate)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        # for feature_weights - st
        bottom_entropy = self.observed_value_probabilities * tf.math.log(self.observed_value_probabilities)
        bottom_entropy = tf.reduce_sum(bottom_entropy)
        self.feature_weight_bottom_entropy = bottom_entropy

        # overall minimum
        self.general_achieved_minimum_params = None

    def calculate_feature_weights(self, probabilities):
        # top = sum over i=1^n Pr[x_i^m | z] * Pr[z] * log( Pr[x_i^m | z] / Pr[x_i^m])
        # bottom = norm * sum over i=1^n Pr[x_i^m] * log Pr[x_i^m]
        bottom = self.feature_weight_bottom_entropy
        x_mi_probas = tf.transpose(self.observed_value_probabilities)
        x_mi_probas = tf.expand_dims(x_mi_probas, axis=-1)
        top = probabilities * self.new_priors

        # top = top * tf.math.log(probabilities / x_mi_probas)
        top = top * tf.math.log(tf.minimum(1, probabilities / x_mi_probas))
        top = tf.reduce_sum(top, axis=1)  # (m, z)
        weights = top / bottom

        # force the weights to sum to m*z
        Z = tf.reduce_sum(weights) / (self.n_z * self.n_m)
        weights = weights / Z
        weights = tf.expand_dims(weights, axis=1)

        if tf.math.reduce_any(tf.math.is_nan(weights)):
            print("NAN IN WEIGHTS!")

        return weights

    def overflow_sum(self, probabilities):
        sum_probabilities = tf.transpose(tf.reduce_sum(probabilities, axis=-1))
        diff = sum_probabilities - self.observed_value_probabilities
        diff = tf.nn.relu(diff)  # the value being below is okay
        return tf.reduce_sum(diff)

    def func_to_optimize(self, variables):
        # observed_values, observed_value_probabilities,
        # are of shape (n, m)

        # get_probabilities of shape ()
        proba_list = []
        for m in range(self.n_m):
            m_params = [[self.distributions_to_use[m][z], params] for z, params in zip(range(self.n_z),
                                                                                       variables[m])]
            m_probas = tf.stack(
                [self.metric_models[m].calculate_metric_probability(z,
                                                                    m_params=m_params,
                                                                    m_observed_values=self.observed_values[:, m]) for
                 z, params in zip(range(self.n_z), variables[m])]
            )
            m_probas = tf.transpose(m_probas) + self.pseudocount
            proba_list.append(m_probas)

        probabilities = tf.stack(proba_list)  # shape (m, n, z)

        if tf.math.reduce_any(tf.math.is_nan(probabilities)):
            print("NAN IN PROBABILITIES!")
        log_probabilities = tf.math.log(probabilities)

        if tf.math.reduce_any(tf.math.is_nan(log_probabilities)):
            print("NAN IN LOG-PROBABILITIES!")

        feature_weights = self.calculate_feature_weights(probabilities)  # shape (m, z)

        to_sum = feature_weights * log_probabilities
        to_sum = to_sum * self.qs
        objective = - tf.reduce_sum(to_sum)  # to make minimizer from maximizer
        constraint = self.overflow_sum(probabilities) * self.penalty
        return objective + constraint

    def optimize(self):
        last_seen_loss = np.inf

        general_achieved_minimum_objective = np.inf
        self.general_achieved_minimum_params = None

        for i in range(self.iterations):
            with tf.GradientTape() as tape:
                # Record the operations for automatic differentiation
                flat_variables = tf.nest.flatten(self.variables)
                tape.watch(flat_variables)

                loss = self.func_to_optimize(self.variables)

            # store the achieved minimum
            if loss.numpy() < general_achieved_minimum_objective:
                general_achieved_minimum_objective = loss.numpy()
                self.general_achieved_minimum_params = [[tf.constant(var.numpy()) for var in sublist] for sublist in self.variables]

            # Compute gradients
            gradients = tape.gradient(loss, flat_variables)
            self.optimizer.apply_gradients(zip(gradients, flat_variables))

            # Reporting
            if i % self.report_step == 0:
                if np.abs(last_seen_loss - loss.numpy()) < self.tolerance:
                    print(
                        f"{self.ident}:\tChange from the last report is lower than {self.tolerance}, loss: {loss.numpy()}, ending run")
                    break
                last_seen_loss = loss.numpy()
                print(f"{self.ident}:\tOptimization iteration {i}, Loss: {loss.numpy()}")
        # return loss
        return general_achieved_minimum_objective

    def get_optimized_parameters(self):
        return self.general_achieved_minimum_params

    def get_feature_weights(self):
        variables = self.variables
        proba_list = []
        for m in range(self.n_m):
            m_params = [[self.distributions_to_use[m][z], params] for z, params in zip(range(self.n_z),
                                                                                       variables[m])]
            m_probas = tf.stack(
                [self.metric_models[m].calculate_metric_probability(z,
                                                                    m_params=m_params,
                                                                    m_observed_values=self.observed_values[:, m]) for
                 z, params in zip(range(self.n_z), variables[m])]
            )
            m_probas = tf.transpose(m_probas) + self.pseudocount
            proba_list.append(m_probas)

        probabilities = tf.stack(proba_list)  # shape (m, n, z)

        if tf.math.reduce_any(tf.math.is_nan(probabilities)):
            print("NAN IN PROBABILITIES!")

        feature_weights = self.calculate_feature_weights(probabilities)  # shape (m, z)
        return tf.squeeze(feature_weights, axis=[1])


if __name__ == '__main__':
    input_paths = ["test_data/OPTIM_new_priors.tfb",  # tensorflow binary
                   "test_data/OPTIM_observed_value_probas.tfb",
                   "test_data/OPTIM_observed_values.tfb",
                   "test_data/OPTIM_qs.tfb"]

    tensors = []
    for binary_file_path in input_paths:
        binary_content = tf.io.read_file(binary_file_path)
        tensor = tf.io.parse_tensor(binary_content, out_type=tf.float64)
        tensors.append(tensor)

    new_priors, observed_value_probas, observed_values, qs = tensors

    from initialize_metrics import create_metric_instance

    # metric = create_metric_instance(3, "gc", dict(), dict())
    # metric_model = metric.probability_model
    # m_params = [metric_model.fill_params(z, observed_values) for z in range(3)]

    # m = 0
    # optimizer = NumericalOptimizer(qs, observed_values[:, m], observed_value_probas[:, m], new_priors, metric_model,
    #                                10000, 0.05, 0.99, "test",
    #                                {0: "uniform",
    #                                 1: "gaussian-mixture",
    #                                 2: "exponential"})
    # final_loss = optimizer.optimize()

    # gc distributions
    distributions_0 = {0: "uniform",
                       1: "exponential",
                       2: "exponential"}
    # pair-gc distributions
    distributions_1 = {0: "uniform",
                       1: "exponential",
                       2: "exponential"}

    metrics = [create_metric_instance(3, "gc", dict(), dict()),
               create_metric_instance(3, "pair", dict(), dict())]
    metric_models = [metric.probability_model for metric in metrics]

    optimizer = WeightedNumericalOptimizer(qs,
                                           observed_values,
                                           observed_value_probas,
                                           new_priors,
                                           metric_models,
                                           10000, 0.01,
                                           0.95, "test",
                                           {0: distributions_0, 1: distributions_1},
                                           report_step=1000,
                                           )
    final_loss = optimizer.optimize()
    params = optimizer.get_optimized_parameters()
    feature_weights = optimizer.get_feature_weights()

    print(f"done, final loss: {final_loss}")
