import pickle
import os
import numpy as np
import pandas as pd


def analyze_results(taskids, num_reps, base_save_folder):
    all_scores = {}
    for t, taskid in enumerate(taskids):
        train_accuracies = []
        test_accuracies = []
        num_successes = 0
        num_failures = 0

        for r in range(num_reps):
            save_folder = f"{base_save_folder}/{taskid}/Rep_{r}"
            result_file = f"{save_folder}/scores.pkl"
            failed_file = f"{save_folder}/failed.pkl"
            
            if os.path.exists(result_file):
                with open(result_file, "rb") as f:
                    scores = pickle.load(f)
                train_accuracies.append(scores['train_accuracy'])
                test_accuracies.append(scores['test_accuracy'])
                num_successes += 1
            elif os.path.exists(failed_file):
                num_failures += 1
                with open(failed_file, "rb") as f:
                    failure_info = pickle.load(f)
                print(f"Task {taskid} Rep {r} failed")
                print(failure_info['trace'])
            else:
                continue  # Neither success nor failure recorded

        # Compute averages
        avg_train_acc = np.mean(train_accuracies) if train_accuracies else 0.0
        avg_test_acc = np.mean(test_accuracies) if test_accuracies else 0.0

        all_scores[taskid] = {
            'avg_train_accuracy': avg_train_acc,
            'avg_test_accuracy': avg_test_acc,
            'num_successes': num_successes,
            'num_failures': num_failures
        }

        print(f"Task {taskid}:")
        print(f"  Average Train Accuracy: {avg_train_acc:.4f}")
        print(f"  Average Test Accuracy: {avg_test_acc:.4f}")
        print(f"  Number of Successes: {num_successes}")
        print(f"  Number of Failures: {num_failures}")
        print()

    return all_scores



if __name__ == "__main__":
    taskids=taskids = [190412, 146818, 359955, 168757, 359956, 359958, 359962, 190137, 168911, 359965, 190411, 146820, 359968, 359975, 359972, 168350, 359971]
    
    num_reps = 10
    base_save_folder = "Results/tpot"
    
    results = analyze_results(taskids, num_reps, base_save_folder)

    # Compare against results from Data/Raw_OpenML_Suite_271_Binary_Classification/tasks_summary.csv
    random_runs_df = pd.read_csv("Data/Raw_OpenML_Suite_271_Binary_Classification/tasks_summary.csv")

    # read relevant columns: task_id, DT,ET,GB,KSVC,LSGD,LSVC,RF
    relevant_columns = ['task_id', 'DT', 'ET', 'GB', 'KSVC', 'LSGD', 'LSVC', 'RF']
    random_runs_df = random_runs_df[relevant_columns]

    # make a dictionary where each element is task_id: {"max": max(random_runs_df[DT], random_runs_df[ET], ...), ], "avg": avg(...)}
    random_runs_summary = {}
    for index, row in random_runs_df.iterrows():
        task_id = row['task_id']
        accuracies = [row['DT'], row['ET'], row['GB'], row['KSVC'], row['LSGD'], row['LSVC'], row['RF']]
        max_acc = np.max(accuracies)
        avg_acc = np.mean(accuracies)
        random_runs_summary[task_id] = {
            'max_accuracy': max_acc,
            'avg_accuracy': avg_acc
        }
    # Compare results
    for taskid in taskids:
        tpot_test_acc = results[taskid]['avg_test_accuracy']
        random_max_acc = random_runs_summary[taskid]['max_accuracy']
        random_avg_acc = random_runs_summary[taskid]['avg_accuracy']
        print(f"Task {taskid} Comparison:")
        print(f"  TPOT Average Test Accuracy: {tpot_test_acc:.4f}")
        print(f"  Random Runs Max Accuracy: {random_max_acc:.4f}")
        print(f"  Random Runs Average Accuracy: {random_avg_acc:.4f}")
        print()