import pickle
import os
import numpy as np
import pandas as pd


def analyze_results(taskids, num_reps, base_save_folder):
    all_scores = {}
    for taskid in taskids:
        train_scores = []
        test_scores = []
        metric = None
        num_successes = 0
        num_failures = 0

        for r in range(num_reps):
            save_folder = f"{base_save_folder}/{taskid}/Rep_{r}"
            result_file = f"{save_folder}/scores.pkl"
            failed_file = f"{save_folder}/failed.pkl"
            
            if os.path.exists(result_file):
                with open(result_file, "rb") as f:
                    scores = pickle.load(f)
                train_scores.append(scores['train_score'])
                test_scores.append(scores['test_score'])
                metric = scores['metric']
                num_successes += 1
            elif os.path.exists(failed_file):
                num_failures += 1
                with open(failed_file, "rb") as f:
                    failure_info = pickle.load(f)
                print(f"Task {taskid} Rep {r} failed")
                print(failure_info['trace'])
            else:
                continue  # Neither success nor failure recorded

        avg_train_score = np.mean(train_scores) if train_scores else 0.0
        avg_test_score = np.mean(test_scores) if test_scores else 0.0

        all_scores[taskid] = {
            'avg_train_score': avg_train_score,
            'avg_test_score': avg_test_score,
            'metric': metric,
            'num_successes': num_successes,
            'num_failures': num_failures
        }

        print(f"Task {taskid}:")
        print(f"  Metric: {metric or 'N/A'}")
        print(f"  Average Train Score: {avg_train_score:.4f}")
        print(f"  Average Test Score: {avg_test_score:.4f}")
        print(f"  Number of Successes: {num_successes}")
        print(f"  Number of Failures: {num_failures}")
        print()

    return all_scores



if __name__ == "__main__":
    data_directory = "Data/Raw_OpenML_Suite_271_Classification"
    summary = pd.read_csv(os.path.join(data_directory, "tasks_summary.csv"))
    taskids = summary['task_id'].astype(int).tolist()
    num_reps = 10
    base_save_folder = "Results/tpot"
    analyze_results(taskids, num_reps, base_save_folder)
