import sys

import numpy as np
import tensorflow as tf
from argmax_calculation import NumericalOptimizer, WeightedNumericalOptimizer
from itertools import product


class BasicModelEnsemble:
    def __init__(self, metric_models, observed_values, n_m, n_z,
                 max_iterations, learning_rate, decay_rate, report_step, identificator):
        self.trained = False
        self.final_parameters = None
        self.final_priors = None
        self.models = metric_models
        self.observed_values = observed_values
        self.n_m = n_m
        self.n_z = n_z
        self.identificator = identificator

        # optimizer params
        self.optimizer_max_iterations = max_iterations
        self.optimizer_learning_rate = learning_rate
        self.optimizer_decay_rate = decay_rate
        self.optimizer_report_step = report_step

        def calculate_frequency(column):
            y, idx, count = tf.unique_with_counts(column)
            freq = tf.gather(count, idx)
            return freq

        freqs = tf.map_fn(calculate_frequency, tf.transpose(observed_values)) / len(observed_values)
        self.observed_value_probabilities = tf.transpose(freqs)

    def calculate_log_probabilities(self, z, parameters, observed_values):
        a = []
        for m in range(len(self.models)):
            x = self.models[m].calculate_metric_probability(z,
                                                            parameters[m],
                                                            observed_values[:, m])
            a.append(x)
        probabilities = tf.stack(a, axis=1)
        log_probas = tf.math.log(probabilities)
        return log_probas

    def calculate_probability(self, z, parameters):
        log_probas = self.calculate_log_probabilities(z, parameters, self.observed_values)
        return tf.exp(tf.reduce_sum(log_probas, axis=1))

    def argmax_new_params(self, qs, new_priors):
        # argmax sum_m sum_z sum_i qs[z]_i log Pr[x_i^m | z] for each allowed probability
        # subj to Pr[x_i^m] >= sum_z Pr[x_i^m | z]
        new_params = []
        new_achieved_func = 0

        for m in range(self.n_m):
            # get information on the allowed distributions
            print(f"{self.identificator}:\ttraining metric {m}")
            metric_proba_model = self.models[m]
            all_dists = [metric_proba_model.get_allowed_distributions(z) for z in range(self.n_z)]

            best_achieved_m_objective = np.inf
            best_m_params = None
            for dists in product(*all_dists):
                print(f"{self.identificator}:\ttraining for {dists}")
                distributions_to_use = {z: d for z, d in enumerate(dists)}

                optimizer = NumericalOptimizer(qs,
                                               self.observed_values[:, m],
                                               self.observed_value_probabilities[:, m],
                                               new_priors,
                                               metric_proba_model,
                                               self.optimizer_max_iterations,
                                               self.optimizer_learning_rate,
                                               self.optimizer_decay_rate,
                                               self.identificator,
                                               distributions_to_use)
                objective = optimizer.optimize()  # minimizing
                new_m_params = optimizer.get_optimized_parameters()

                if objective < best_achieved_m_objective:
                    best_achieved_m_objective = objective
                    best_m_params = [[dists[z], new_m_params[z]] for z in range(self.n_z)]
                    print(f"{self.identificator}:\t{dists} are best so far ")

            new_params.append(best_m_params)
            new_achieved_func += best_achieved_m_objective

        return new_params, new_achieved_func

    def finalize_training(self, final_prior_probas, final_parameters):
        if self.trained:
            raise Exception("Cannot finalize, model was already trained.")
        self.final_parameters = final_parameters
        self.final_priors = final_prior_probas
        self.trained = True

    def calculate_final_probability(self, unseen_observed_values):
        assert self.trained

        mismatch_proba = tf.exp(self.calculate_log_probabilities(0,
                                                                 self.final_parameters,
                                                                 unseen_observed_values))
        match_probas = []
        for z in range(1, self.n_z):
            match_probas.append(
                tf.exp(self.calculate_log_probabilities(z,
                                                        self.final_parameters,
                                                        unseen_observed_values))
            )
        return mismatch_proba, match_probas


class WeightedModelEnsemble(BasicModelEnsemble):
    def __init__(self, metric_models, observed_values, n_m, n_z,
                 max_iterations, learning_rate, decay_rate, report_step, identificator):
        super().__init__(metric_models, observed_values, n_m, n_z,
                         max_iterations, learning_rate, decay_rate, report_step, identificator)
        self.weights = tf.ones((n_m, n_z), dtype=tf.float64) * 2

    def calculate_probability(self, z, parameters):
        log_probas = self.calculate_log_probabilities(z, parameters, self.observed_values)

        weighted_log_probas = log_probas * self.weights[:, z]

        log_probas_sum = tf.reduce_sum(weighted_log_probas, axis=1)
        return tf.exp(log_probas_sum)

    def argmax_new_params(self, qs, new_priors):
        all_dists_dicts = []
        for m in range(self.n_m):
            metric_proba_model = self.models[m]
            all_dists = [metric_proba_model.get_allowed_distributions(z) for z in range(self.n_z)]

            x = []
            for dists in product(*all_dists):
                distributions_to_use = {z: d for z, d in enumerate(dists)}
                x.append(distributions_to_use)
            all_dists_dicts.append(x)

        best_achieved_m_objective = np.inf
        best_params = None
        best_weights = None
        for dists in product(*all_dists_dicts):
            print(f"{self.identificator}:\ttraining for {dists}")
            distributions_to_use = {m: d for m, d in enumerate(dists)}
            optimizer = WeightedNumericalOptimizer(qs,
                                                   self.observed_values,
                                                   self.observed_value_probabilities,
                                                   new_priors,
                                                   self.models,
                                                   self.optimizer_max_iterations,
                                                   self.optimizer_learning_rate,
                                                   self.optimizer_decay_rate,
                                                   self.identificator,
                                                   distributions_to_use
                                                   )
            objective = optimizer.optimize()  # minimizing
            new_m_params = optimizer.get_optimized_parameters()

            if objective < best_achieved_m_objective:
                best_achieved_m_objective = objective
                best_params = [[[distributions_to_use[m][z], new_m_params[m][z]] for z in range(self.n_z)] for m in
                               range(self.n_m)]
                best_weights = optimizer.get_feature_weights()
                print(f"{self.identificator}:\t{dists} are best so far ")

        new_params = best_params
        new_achieved_func = best_achieved_m_objective
        self.weights = best_weights
        return new_params, new_achieved_func

    def calculate_final_probability(self, unseen_observed_values):
        assert self.trained

        mis_log_proba = self.calculate_log_probabilities(0,
                                                         self.final_parameters,
                                                         unseen_observed_values)
        mis_log_proba = mis_log_proba * self.weights[:, 0]
        mis_log_proba = tf.reduce_sum(mis_log_proba, axis=1)
        mismatch_proba = tf.exp(mis_log_proba)

        match_probas = []
        for z in range(1, self.n_z):
            z_log_proba = self.calculate_log_probabilities(z,
                                                           self.final_parameters,
                                                           unseen_observed_values)
            z_log_proba = z_log_proba * self.weights[:, z]
            match_probas.append(
                tf.exp(tf.reduce_sum(z_log_proba, axis=1))
            )
        return mismatch_proba, match_probas


class EMOptimizer:
    def __init__(self, observed_values,
                 no_matched_models,
                 metric_models,
                 use_metric_weighting=False,
                 max_step=10,
                 target_func_tolerance=0.01,
                 pseudocount=1e-10,
                 alpha=0.1,
                 identificator="",
                 max_iterations=10000,
                 learning_rate=0.1,
                 decay_rate=0.9,
                 report_step=100,
                 reporter_file_name=None,
                 sep=';', minisep=','):
        # self.observed_values = observed_values
        self.n_m = observed_values.shape[1]
        self.n_z = no_matched_models + 1
        self.N = len(observed_values)
        self.sep = sep
        self.minisep = minisep

        if use_metric_weighting:
            self.model_ensemble = WeightedModelEnsemble(metric_models, observed_values,
                                                        n_m=self.n_m,
                                                        n_z=self.n_z,
                                                        max_iterations=max_iterations,
                                                        learning_rate=learning_rate,
                                                        decay_rate=decay_rate,
                                                        report_step=report_step,
                                                        identificator=identificator)
        else:
            self.model_ensemble = BasicModelEnsemble(metric_models, observed_values,
                                                     n_m=self.n_m,
                                                     n_z=self.n_z,
                                                     max_iterations=max_iterations,
                                                     learning_rate=learning_rate,
                                                     decay_rate=decay_rate,
                                                     report_step=report_step,
                                                     identificator=identificator)

        self.max_step = max_step
        self.target_func_tolerance = target_func_tolerance
        self.identificator = identificator
        self.pseudocount = pseudocount
        self.alpha = alpha
        self.observed_values = observed_values
        if reporter_file_name is not None:
            self.reporter_file = open(reporter_file_name + self.identificator, mode='w')
        else:
            self.reporter_file = sys.stdout

    def __e_step(self, prior_probas, theta_params):
        # top[z] =  PRIOR[z] * Pr[x | z, params] for every z
        # bottom = sum PRIOR[z] * Pr[x | z, params] over all z
        tops = tf.stack([self.model_ensemble.calculate_probability(z, theta_params) * prior_probas[z] for z in
                         range(self.n_z)],
                        axis=1)
        bottom = tf.reduce_sum(tops, axis=1)

        qs = tf.transpose(tops) / bottom
        return tf.transpose(qs)

    def __m_step(self, qs):
        # calculate new prior probabilities
        new_prior_probas = []
        total = 0
        for z in range(self.n_z):
            p = np.mean(qs[z])
            total += p
            new_prior_probas.append(p)
        new_prior_probas = tf.constant([x / total for x in new_prior_probas])

        # get argmax sum_i=1^n sum_m sum_z q_i[z] * w(m,z) * log Pr[x_i^(m) | z]
        new_theta_params, achieved_objective = self.model_ensemble.argmax_new_params(qs, new_prior_probas)
        return new_prior_probas, new_theta_params, achieved_objective

    def __init_report(self, init_priors, init_theta):
        # initial
        columns = ["identificator", "objective"]
        for z in range(self.n_z):
            columns.append(f"prior:{z}")
        for m in range(self.n_m):
            for z in range(self.n_z):
                columns.append(f"theta:{m}:{z}:dist")
                columns.append(f"theta:{m}:{z}:params")
        print(self.sep.join(columns), file=self.reporter_file)

    def __report(self, new_priors, new_theta, achieved_objective):
        report_line = [self.identificator, "{:.2f}".format(achieved_objective)]
        for z in range(self.n_z):
            report_line.append("{:.2f}".format(new_priors[z].numpy()))
        for m in range(self.n_m):
            for z in range(self.n_z):
                report_line.append(new_theta[m][z][0])
                m_params = new_theta[m][z][1]
                report_line.append(self.minisep.join(["{:.2f}".format(x) for x in m_params.numpy()]))
        print(self.sep.join(report_line), file=self.reporter_file)

    def optimize(self):
        prior_probas = tf.constant([1 - self.alpha,
                                    *[self.alpha / (self.n_z - 1) for _ in range(self.n_z - 1)]],
                                   dtype=tf.float64)  # array of length n_z
        theta_params = [None for m in range(self.n_m)]
        # array of m_params
        # m_params is a list of z sets of params
        last_objective_value = np.inf
        self.__init_report(prior_probas, theta_params)

        for em_i in range(self.max_step):
            print(f"{self.identificator}:\tEM-ITERATION {em_i}")
            qs = self.__e_step(prior_probas, theta_params)
            new_prior_probas, new_theta_params, achieved_objective = self.__m_step(qs)
            self.__report(new_prior_probas, new_theta_params, achieved_objective)

            # assign new values
            prior_probas = new_prior_probas
            theta_params = new_theta_params

            # check for convergence
            if np.abs(last_objective_value - achieved_objective) < self.target_func_tolerance:
                print(f"{self.identificator}:\tEM-CONVERGENCE in {em_i}")
                break
            last_objective_value = achieved_objective

        self.model_ensemble.finalize_training(prior_probas, theta_params)
        self.reporter_file.close()
        # TODO do not close if stdout
        return self.model_ensemble
