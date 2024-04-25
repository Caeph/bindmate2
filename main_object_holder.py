import time

from input_parsing import load_bed_input, load_fasta_input, read_unique_kmers
from initialize_metrics import initialize_functions
from seq_to_seq_score_estimation import *
from kmer_to_kmer_score_estimation import *

import numpy as np

from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
from joblib import cpu_count
import os

default_data_bootstrap_no = int(1e3)  # TODO reasonable data bootstrap default


class PairingResults:
    def __init__(self, unique_kmers, kmer_combinations, matched_probas, probas_0, probas_1, mapped_kmers):
        self.unique_kmers = unique_kmers  # numpy array
        self.available_combinations = pd.DataFrame(kmer_combinations, columns=["kmer_i1", "kmer_i2"]
                                                   ).reset_index().rename(
            columns={"index": "combination_index"})  # indexes, pandas df
        # self.probas_1 = probas_1  # np array
        self.matched_probas = matched_probas  # list of np array

        self.probas_1 = probas_1  # np array, matched proba summed - OR logic
        self.probas_0 = probas_0  # np array
        self.mapped_kmers_df = mapped_kmers[["input_sequence_index", "kmer_index"]]  # pandas df

    def get_all_seq_indices(self):
        return self.mapped_kmers_df['input_sequence_index'].unique()

    def save(self, path):
        # TODO write ranks, write all calculated probabilities
        os.makedirs(path, exist_ok=True)
        self.available_combinations.to_csv(os.path.join(path, "available_combinations.csv.gz"),
                                           index=False)
        self.mapped_kmers_df.to_csv(os.path.join(path, "mapped_kmers_df.csv.gz"),
                                    index=False)
        for array, core_filename in zip(
                [self.probas_1, self.probas_0],
                ["probas_1", "probas_0"]
        ):
            np.save(os.path.join(path, core_filename + ".npy"), array)

        np.savetxt(os.path.join(path, "unique_kmers.txt"), self.unique_kmers, fmt="%s")

    # @staticmethod
    # def load(path):
    #     unique_kmers = np.loadtxt(os.path.join(path, "unique_kmers.txt"), dtype=str)
    #     kmer_combinations = pd.read_csv(os.path.join(path, "available_combinations.csv.gz"))
    #     probas_1 = np.load(os.path.join(path, "probas_1.npy"))
    #     probas_0 = np.load(os.path.join(path, "probas_0.npy"))
    #     mapped_kmers = pd.read_csv(os.path.join(path, "mapped_kmers_df.csv.gz"))
    #     # TODO matched probas
    #     return PairingResults(unique_kmers, kmer_combinations, probas_1, probas_0, mapped_kmers)


# TODO create another version that does not come to a single thread after optimization but after preselection


class PairingBasedSimilarityCalculator:
    def __init__(self,
                 k,
                 metrics,
                 matched_models,
                 preselection_part=0.5,
                 additional_info_on_metrics=None,
                 background_info=None,
                 data_bootstrap_no=None,
                 feature_bootstrap_no=None,
                 bootstrap_runs=1,
                 max_em_iterations=10,
                 use_feature_weighting=True,
                 threads=2,
                 reporter_file_name=None,
                 max_iterations=10000,
                 learning_rate=0.01,
                 decay_rate=0.95,
                 report_step=100
                 ):
        # set up parameters
        self.k = k
        if threads == -1:
            self.threads = cpu_count()
        else:
            self.threads = threads
        self.metrics = metrics
        self.additional_info_on_metrics = additional_info_on_metrics
        self.matched_models = matched_models
        self.background_info = background_info

        self.preselection_part = preselection_part

        self.bootstrap_runs = bootstrap_runs
        if data_bootstrap_no is not None:
            self.data_bootstrap_no = data_bootstrap_no
        else:
            self.data_bootstrap_no = default_data_bootstrap_no

        if feature_bootstrap_no is not None:
            self.feature_bootstrap_no = feature_bootstrap_no
        else:
            self.feature_bootstrap_no = len(metrics)  # use everything
        self.max_em_iterations = max_em_iterations
        self.use_feature_weighting = use_feature_weighting
        self.em_params_reporter_filename = reporter_file_name
        self.max_optimizer_iterations = max_iterations
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.report_step = report_step

    def fit_predict_fasta(self, fasta_filename):
        # read fasta and prep input
        # analyze
        self.__write_descriptions(fasta_filename)
        sequence_df = load_fasta_input(fasta_filename)
        return self.__fit(sequence_df)

    def __write_descriptions(self, input_filename):
        # TODO write recap of used params
        ...

    def fit_predict_bed(self, bed_filename, source_fasta):
        # read bed and prep input
        # analyze
        self.__write_descriptions([bed_filename, source_fasta])
        sequence_df = load_bed_input(bed_filename, source_fasta)
        similarities = self.__fit(sequence_df)
        return similarities

    def __get_observed_values(self, kmer_combinations, full_metrics):
        rank_results = []
        for metric in full_metrics:
            start = time.time()
            metric_values = calculate_all_to_all_metric(kmer_combinations, metric)
            print(f"Metric {metric.name} values calculated: {time.time() - start}")
            start = time.time()

            rank_values = translate_metric_to_rank(metric_values, metric.get_type())

            rank_results.append(rank_values)
            print(f"Metric {metric.name} ranks calculated: {time.time() - start}")

        pairwise_ranks = tf.stack(rank_results, axis=1)
        return pairwise_ranks

    def __preselect(self, full_probas_0, full_probas_1, match_probas, kmer_combinations):
        # TODO more flexibility based on match_probas
        part = self.preselection_part

        if part >= 1:
            return np.arange(0, len(full_probas_0))

        without_symmetry = kmer_combinations[:, 0] >= kmer_combinations[:, 1]

        probas_0 = full_probas_0[without_symmetry]
        probas_1 = full_probas_1[without_symmetry]
        no = int(part * len(probas_0))
        one_thr = probas_1[tf.argsort(-probas_1)[no]]

        chosen = tf.where((full_probas_1 >= one_thr) & without_symmetry)

        return tf.squeeze(chosen)

    def __optimize(self, pairwise_ranks, full_metrics):
        all_trained_models = []
        for run_i in range(self.bootstrap_runs):
            trained_models = inner_optimization_run((pairwise_ranks,
                                                     full_metrics,
                                                     f"{run_i}",
                                                     self.data_bootstrap_no,
                                                     self.feature_bootstrap_no,
                                                     self.matched_models,
                                                     self.use_feature_weighting,
                                                     self.max_em_iterations,
                                                     self.em_params_reporter_filename,
                                                     self.max_optimizer_iterations,
                                                     self.learning_rate,
                                                     self.decay_rate,
                                                     self.report_step
                                                     ))
            all_trained_models.append(trained_models)
        return all_trained_models

    def __combine_probabilities_from_bootstrap(self, all_trained_models, pairwise_ranks):
        # all trained model is a list of what the inner_optimization_run returns: trained_models, feature_indices
        # todo implement, add more flexibility with the combining logic
        bootstrapped_mismatch = []
        bootstrapped_match = []
        for trained_model_ensemble, source_features in all_trained_models:
            strap_mismatch_proba, strap_match_probas = trained_model_ensemble.calculate_final_probability(
                tf.gather(pairwise_ranks, indices=source_features, axis=1))  # a model ensemble
            bootstrapped_mismatch.append(strap_mismatch_proba)
            bootstrapped_match.append(strap_match_probas)

        if len(all_trained_models) == 1:
            mismatch_proba, match_probas = bootstrapped_mismatch[0], bootstrapped_match[0]
        else:
            # TODO combine bootstrapped probabilities
            # Todo store the bootstrapped singletons for future reference
            raise NotImplementedError("TODO combine probabilities from bootstrapping")

        total_match_proba = np.zeros_like(mismatch_proba)
        for model_proba in match_probas:
            total_match_proba = total_match_proba + model_proba
        return mismatch_proba, match_probas, total_match_proba

    def __calculate_kmer_to_kmer_similarities(self, unique_kmers, full_metrics):
        start = time.time()
        kmer_combinations = get_kmer_combinations(unique_kmers)  # indices only
        print(f"Unique kmer combinations created: {time.time() - start}")

        # calculate ranks
        pairwise_ranks = self.__get_observed_values(kmer_combinations, full_metrics)

        # get kmer similarity probabilities
        all_trained_models = self.__optimize(pairwise_ranks, full_metrics)

        # TODO logic of probability mashups
        mismatch_proba, match_probas, total_match_proba = self.__combine_probabilities_from_bootstrap(
            all_trained_models, pairwise_ranks)

        # preselect
        selected_indices = self.__preselect(mismatch_proba,
                                            total_match_proba,
                                            match_probas,
                                            kmer_combinations)

        selected_kmer_combinations = tf.gather(kmer_combinations, selected_indices, axis=0)
        selected_mismatch_proba = tf.gather(mismatch_proba, selected_indices)
        selected_match_proba = [tf.gather(model_proba, selected_indices) for model_proba in match_probas]
        selected_total_match_proba = tf.gather(total_match_proba, selected_indices)

        return selected_kmer_combinations, selected_match_proba, selected_mismatch_proba, selected_total_match_proba

    def __calculate_seq_to_seq_similarities(self, kmer_match_proba_obj):
        seq_ids = kmer_match_proba_obj.get_all_seq_indices()
        all_to_all = np.array(np.meshgrid(seq_ids, seq_ids))

        todo = np.triu(all_to_all)  # applied to final two
        todo = np.vstack([
            np.reshape(todo[0, :, :], (-1, 1)).flatten(),
            np.reshape(todo[1, :, :], (-1, 1)).flatten()
        ]).T
        todo = np.unique(todo, axis=0)
        todo = todo[todo[:, 0] != todo[:, 1]]  # don't compare the same sequences
        todo = pd.DataFrame(todo, columns=['seq1_index', 'seq2_index'])

        # TO SPEED UP - ONLY KEEPING I > J PAIRS OF SEQUENCES
        kmer_combinations = kmer_match_proba_obj.available_combinations
        seq_to_kmer = kmer_match_proba_obj.mapped_kmers_df

        # sequence pairs with mapped kmers
        print("Mapping sequences and kmer probabilities")
        X = pd.merge(todo, seq_to_kmer, left_on='seq1_index', right_on='input_sequence_index'
                     ).drop(columns=['input_sequence_index']).rename(columns={"kmer_index": "kmer_i1"})
        X = pd.merge(X, seq_to_kmer, left_on='seq2_index', right_on='input_sequence_index'
                     ).drop(columns=['input_sequence_index']).rename(columns={"kmer_index": "kmer_i2"})

        # filter to contain only available kmer combinations
        X = pd.merge(X, kmer_combinations)
        X['probability'] = X["combination_index"].apply(lambda ci: kmer_match_proba_obj.probas_1[ci])

        print("Creating graphs")
        graphs = []
        for item in tqdm(X.groupby(by=['seq1_index', 'seq2_index'])):
            name, group = item
            G = create_bipartite_graph(group)
            G.graph['name'] = name  # s1, s2 indices
            graphs.append(G)

        print("Calculating distance as perfect matching")

        if self.threads == 1:
            results = []
            for item in tqdm(graphs):
                r = find_best_pairing(item)
                results.append(r)
        else:
            no = len(graphs)
            with Pool(self.threads) as pool:
                results = tqdm(pool.imap(find_best_pairing, graphs),
                               total=no)
                results = list(results)

        return results

    def __fit(self, sequence_df):
        # init kmers
        unique_kmers, kmers_mapped_to_sqs = read_unique_kmers(sequence_df, self.k)
        # init functions
        full_metrics = initialize_functions(k=self.k,
                                            no_matched_models=self.matched_models,
                                            unique_kmers=unique_kmers,
                                            metrics=self.metrics,
                                            background_info=self.background_info,
                                            additional_info_on_metrics=self.additional_info_on_metrics)
        # kmer to kmer scores
        selected_kmer_combinations, selected_match_proba, selected_mismatch_proba, selected_total_match_proba = self.__calculate_kmer_to_kmer_similarities(
            unique_kmers, full_metrics)
        kmer_to_kmer_results = PairingResults(
            unique_kmers=unique_kmers,
            kmer_combinations=selected_kmer_combinations.numpy(),
            matched_probas=[x.numpy() for x in selected_match_proba],
            probas_0=selected_mismatch_proba.numpy(),
            probas_1=selected_total_match_proba.numpy(),
            mapped_kmers=kmers_mapped_to_sqs
        )

        # seq to seq scores - list of objects
        seq_to_seq_results = self.__calculate_seq_to_seq_similarities(kmer_to_kmer_results)

        return seq_to_seq_results, kmer_to_kmer_results
