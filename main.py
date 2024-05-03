from main_object_holder import PairingBasedSimilarityCalculator
from background_normalizer import Normalizer


def main():
    input_file = "test_data/small_unbalanced_test_dataset_randombg.fasta"
    # input_file = "test_data/biodata_CTCF_TP53_l:300_n:10:10.fasta"

    background_info = dict(type="sampled", size=1500, source="background_data/upstream2000.fa")
    sampled_normalizer = Normalizer(background_info, 31)
    metrics_info = {
        "pair": dict(motif_selection=[],
                     background_info=background_info),
        "gc": dict(background_object=sampled_normalizer),
        "probound": dict(selected_motifs=["ATF2", "TEAD1", "EGR1"],
                         taxa="Homo sapiens",
                         background_object=sampled_normalizer
                         ),
        "shape": dict(shape_feature="EP",
                      shape_aggregator="max",
                      background_object=sampled_normalizer)
    }

    calculator = PairingBasedSimilarityCalculator(
        k=31,
        metrics=["gc", "pair", "probound",
            "shape"
                    ],
        matched_models=2,
        additional_info_on_metrics=metrics_info,
        threads=8,
        max_em_iterations=3,
        max_iterations=200,
        decay_rate=0.95,
        reporter_file_name="test_output/em_params_record.log",
    )
    seq_to_seq_results, kmer_to_kmer_results = calculator.fit_predict_fasta(input_file)
    kmer_to_kmer_results.save("test_output/test_run_results")
    print()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
