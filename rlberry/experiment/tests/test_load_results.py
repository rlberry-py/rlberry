import os
import sys
import tempfile

from rlberry.experiment import experiment_generator, load_experiment_results

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


def test_save_and_load():
    TEST_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.argv = [TEST_DIR + "/test_load_results.py"]
    with tempfile.TemporaryDirectory() as tmpdirname:
        sys.argv.append(TEST_DIR + "/params_experiment.yaml")
        sys.argv.append("--parallelization=thread")
        sys.argv.append("--output_dir=" + tmpdirname)
        print(sys.argv)
        for experiment_manager in experiment_generator():
            experiment_manager.fit()
            experiment_manager.save()
        data = load_experiment_results(tmpdirname, "params_experiment")

        assert len(data) > 0
