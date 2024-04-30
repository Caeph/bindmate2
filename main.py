from main_object_holder import PairingBasedSimilarityCalculator
from background_normalizer import Normalizer


def main():
    input_file = "test_data/small_unbalanced_test_dataset_randombg.fasta"
    # input_file = "test_data/biodata_CTCF_TP53_l:300_n:10:10.fasta"

    background_info = dict(type="sampled", size=1500, source="background_data/upstream2000.fa")
    metrics_info = {
        "pair": dict(motif_selection=[],
                     background_info=background_info),
        "gc": dict(background_object=Normalizer(dict(type='random', size=1500), 31))
    }

    calculator = PairingBasedSimilarityCalculator(
        k=31,
        metrics=["gc", "pair"],
        matched_models=2,
        additional_info_on_metrics=metrics_info,
        threads=8,
        max_em_iterations=3,
        max_iterations=200,
        decay_rate=0.95,
        reporter_file_name="test_output/em_params_record.log",
        kmer_results_file_name="test_output/kmer_results.csv"
    )
    seq_to_seq_results, kmer_to_kmer_results = calculator.fit_predict_fasta(input_file)
    kmer_to_kmer_results.save("test_output/test_run_results")
    print()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
