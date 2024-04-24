import numpy as np
import tensorflow as tf
from em_algorithm import *


def calculate_all_to_all_metric(kmer_combinations, metric):
    def calculate(r):
        i1, i2 = r[0], r[1]
        return metric.compare_kmers(i1, i2)

    metric_values = tf.map_fn(calculate, kmer_combinations, dtype=tf.float32)
    return metric_values


def get_kmer_combinations(unique_kmers):
    n = len(unique_kmers)
    indices = tf.range(n)
    tensor_3d = tf.stack(tf.meshgrid(indices, indices, indexing='ij'), axis=-1)
    return tf.reshape(tensor_3d, [-1, 2])


def translate_metric_to_rank(value_array, metric_type):
    # unique_values, idx = tf.unique(tf.reshape(value_array, [-1]))
    unique_values, idx = tf.unique(value_array)

    if metric_type == 'similarity':  # the higher value the more similar
        order = tf.argsort(unique_values, direction='DESCENDING')
    elif metric_type == 'distance':  # the lower value the more similar
        order = tf.argsort(unique_values, direction='ASCENDING')
    else:
        raise NotImplementedError("Unknown metric type.")

    ranks = tf.argsort(order)
    ranked_flat_matrix = tf.gather(ranks, idx)
    return ranked_flat_matrix


def bootstrap(original_ranks, full_metrics, data_bootstrap_no, feature_bootstrap_no):
    data_indices = tf.random.shuffle(tf.range(tf.shape(original_ranks)[0]))[:data_bootstrap_no]
    feature_indices = tf.random.shuffle(tf.range(tf.shape(original_ranks)[1]))[:feature_bootstrap_no]

    selected_rows = tf.gather(original_ranks, data_indices)
    bootstrapped_data = tf.gather(selected_rows, feature_indices, axis=1)

    boostrapped_models = [full_metrics[i].get_model_for_optimization() for i in feature_indices.numpy()]
    return bootstrapped_data, boostrapped_models, feature_indices


def inner_optimization_run(params):
    pairwise_ranks, full_metrics, identificator, data_bootstrap_no, feature_bootstrap_no, matched_models, use_feature_weighting, max_em_iterations, reporter_file_name, \
        max_iterations, learning_rate, decay_rate, report_step = params
    bootstrapped_ranks, bootstrapped_models, feature_indices = bootstrap(pairwise_ranks,
                                                                         full_metrics,
                                                                         data_bootstrap_no,
                                                                         feature_bootstrap_no)
    optimizer = EMOptimizer(observed_values=bootstrapped_ranks,
                            no_matched_models=matched_models,
                            metric_models=bootstrapped_models,
                            use_metric_weighting=use_feature_weighting,
                            max_step=max_em_iterations,
                            identificator=identificator,
                            reporter_file_name=reporter_file_name,
                            max_iterations=max_iterations,
                            learning_rate=learning_rate,
                            decay_rate=decay_rate,
                            report_step=report_step,
                            )
    trained_models = optimizer.optimize()
    return trained_models, feature_indices
