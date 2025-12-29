import pickle
import os
import numpy as np


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
    taskids=[190412, 146818, 359955, 168757, 359956, 359958, 359962, 190137, 168911, 190392,
          189922, 359965, 359966, 359967, 190411, 146820, 359968, 359975, 359972, 168350,
          359973, 190410, 359971, 359988, 359989, 359979, 359980, 359992, 359982, 167120,
          359990, 189354, 360114, 359994]
    
    num_reps = 20
    base_save_folder = "./Results/TPOT_Comparison"
    
    results = analyze_results(taskids, num_reps, base_save_folder)