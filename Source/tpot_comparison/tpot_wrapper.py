import numpy as np
import tpot
import sklearn
from sklearn.metrics import accuracy_score
from Source.tpot_comparison.param_space_conversion import generate_tpot_search_space
from functools import partial
from Source.Base.data_utils import load_data
from typing import List
import os
import time
import random
import pickle
import traceback
import argparse
from Source.Base.data_utils import cv_data_splitter
import ray

def get_splits(task_id: int, rep: int, data_directory: str, split_directory: str):
    rep_dir = os.path.join(split_directory, f"task_{task_id}", f"Replicate_{rep}")
    X_train_raw, _, y_train_raw, _ = load_data(task_id, data_directory, rep_dir)

    # Load CV splits - cv_data_splitter returns ray.put() objects
    # We materialize them immediately here to pass actual numpy arrays to the objective function
    
    # Load all 5 CV folds and materialize Ray references
    X_train_f0, X_val_f0, y_train_f0, y_val_f0 = cv_data_splitter(
        X_train=X_train_raw, y_train=y_train_raw,
        fold_train_path=os.path.join(rep_dir, "fold_train_0.pkl"),
        fold_validate_path=os.path.join(rep_dir, "fold_validate_0.pkl"),
        task_id=task_id, data_dir=data_directory)
    X_train_f0, X_val_f0, y_train_f0, y_val_f0 = ray.get([X_train_f0, X_val_f0, y_train_f0, y_val_f0])
    
    X_train_f1, X_val_f1, y_train_f1, y_val_f1 = cv_data_splitter(
        X_train=X_train_raw, y_train=y_train_raw,
        fold_train_path=os.path.join(rep_dir, "fold_train_1.pkl"),
        fold_validate_path=os.path.join(rep_dir, "fold_validate_1.pkl"),
        task_id=task_id, data_dir=data_directory)
    X_train_f1, X_val_f1, y_train_f1, y_val_f1 = ray.get([X_train_f1, X_val_f1, y_train_f1, y_val_f1])
    
    X_train_f2, X_val_f2, y_train_f2, y_val_f2 = cv_data_splitter(
        X_train=X_train_raw, y_train=y_train_raw,
        fold_train_path=os.path.join(rep_dir, "fold_train_2.pkl"),
        fold_validate_path=os.path.join(rep_dir, "fold_validate_2.pkl"),
        task_id=task_id, data_dir=data_directory)
    X_train_f2, X_val_f2, y_train_f2, y_val_f2 = ray.get([X_train_f2, X_val_f2, y_train_f2, y_val_f2])
    
    X_train_f3, X_val_f3, y_train_f3, y_val_f3 = cv_data_splitter(
        X_train=X_train_raw, y_train=y_train_raw,
        fold_train_path=os.path.join(rep_dir, "fold_train_3.pkl"),
        fold_validate_path=os.path.join(rep_dir, "fold_validate_3.pkl"),
        task_id=task_id, data_dir=data_directory)
    X_train_f3, X_val_f3, y_train_f3, y_val_f3 = ray.get([X_train_f3, X_val_f3, y_train_f3, y_val_f3])
    
    X_train_f4, X_val_f4, y_train_f4, y_val_f4 = cv_data_splitter(
        X_train=X_train_raw, y_train=y_train_raw,
        fold_train_path=os.path.join(rep_dir, "fold_train_4.pkl"),
        fold_validate_path=os.path.join(rep_dir, "fold_validate_4.pkl"),
        task_id=task_id, data_dir=data_directory)
    X_train_f4, X_val_f4, y_train_f4, y_val_f4 = ray.get([X_train_f4, X_val_f4, y_train_f4, y_val_f4])

    X_train_splits = [X_train_f0, X_train_f1, X_train_f2, X_train_f3, X_train_f4]
    X_val_splits = [X_val_f0, X_val_f1, X_val_f2, X_val_f3, X_val_f4]
    y_train_splits = [y_train_f0, y_train_f1, y_train_f2, y_train_f3, y_train_f4]
    y_val_splits = [y_val_f0, y_val_f1, y_val_f2, y_val_f3, y_val_f4]

    return X_train_splits, X_val_splits, y_train_splits, y_val_splits


def custom_objective_function(estimator, X_train_splits, X_val_splits, y_train_splits, y_val_splits):
    scores = []
    for index in range(5):
        X_train, X_val = X_train_splits[index], X_val_splits[index]
        y_train, y_val = y_train_splits[index], y_val_splits[index]

        this_fold_pipeline = sklearn.base.clone(estimator)
        this_fold_pipeline.fit(X_train, y_train)
        y_pred = this_fold_pipeline.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        complexity = tpot.objectives.complexity_scorer(this_fold_pipeline)
        scores.append([acc, complexity])

    return np.mean(scores, axis=0)  # Return mean accuracy and complexity across folds


def tpot_loop_through_tasks(taskids:List[int], data_directory: str, split_directory: str, base_save_folder, num_reps, ga_params):
    
    for t, taskid in enumerate(taskids):
        for r in range(num_reps):
            save_folder = f"{base_save_folder}/{taskid}/Rep_{r}"
            time.sleep(random.random()*5)
            if os.path.exists(save_folder):
                continue
            else:
                os.makedirs(save_folder)

                print("Working on ")
                print(save_folder)

                super_seed = t*1000+r
                print("Super Seed : ", super_seed)
                
                # indicies for train/test split for a given task and replicate
                rep_dir = os.path.join(split_directory, f"task_{taskid}", f"Replicate_{r}")
                X_train, X_test, y_train, y_test = load_data(taskid, data_directory, rep_dir)
                # Get CV splits for custom objective
                X_train_splits, X_val_splits, y_train_splits, y_val_splits = get_splits(taskid, r, data_directory, split_directory)
                
                custom_objective = partial(custom_objective_function, X_train_splits=X_train_splits, X_val_splits=X_val_splits, y_train_splits=y_train_splits, y_val_splits=y_val_splits)
                custom_objective.__name__ = 'custom_objective'
                
                try:
                    linear_est = tpot.TPOTEstimator(
                                            search_space=ga_params["search_space"],
                                            scorers=[],
                                            scorers_weights=[],
                                            other_objective_functions=[custom_objective],
                                            other_objective_functions_weights=[1, -1],
                                            objective_function_names = ['accuracy', 'complexity'],
                                            population_size=ga_params["population_size"],
                                            generations=ga_params["generations"],
                                            classification=True,
                                            max_eval_time_mins=1000000,  # Effectively no time limit per evaluation (~2 years)
                                            n_jobs=ga_params['n_jobs'],
                                            verbose=5,
                                            random_state=super_seed,
                                            )
                    linear_est.fit(X_train, y_train)
                    print("Ending the fitting process. ")
                    accuracy_scorer = sklearn.metrics.get_scorer("accuracy")

                    best_pipeline = linear_est.fitted_pipeline_
                    train_accuracy = accuracy_scorer(best_pipeline, X_train, y_train)
                    train_score = {"train_accuracy": train_accuracy}
                    test_accuracy = accuracy_scorer(best_pipeline, X_test, y_test)
                    test_score = {"test_accuracy": test_accuracy}

                    complexity = tpot.objectives.complexity_scorer(best_pipeline)
                    test_score["complexity"] = complexity

    
                    print("Ending the scoring process. ")

                    this_score = {}
                    this_score.update(train_score)
                    this_score.update(test_score)

                    this_score["pipeline"] = best_pipeline
                    this_score["seed"] = super_seed
                    this_score["run"] = r

                    
                    with open(f"{save_folder}/scores.pkl", "wb") as f:
                        pickle.dump(this_score, f)

                    return           
                
                except Exception as e:
                    trace =  traceback.format_exc()
                    pipeline_failure_dict = {"error": str(e), "trace": trace, "seed": super_seed, "run": r}
                    print("failed on ")
                    print(save_folder)
                    print(e)
                    print(trace)

                    with open(f"{save_folder}/failed.pkl", "wb") as f:
                        pickle.dump(pipeline_failure_dict, f)

                    return
        
    print("all finished")

if __name__ == "__main__":
    # get configs for running EA
    parser = argparse.ArgumentParser(description="Run EA HPO")
    parser.add_argument('--num_cpus', type=int, default=4, help='Number of CPUs to use')
    parser.add_argument('--machine', type=str, default='local', help='Machine identifier')
    args = parser.parse_args()
    num_cpus = args.num_cpus
    machine = args.machine

    classes = 2  # Binary classification
    
    # Search space the same for all TPOT runs
    tpot_search_space = generate_tpot_search_space(classes, num_cpus)

    gp_params_remote = {
        "population_size": 100,
        "generations": 20,
        "search_space": tpot_search_space,
        "n_jobs": num_cpus,
    }
    gp_params_local = {
        "population_size": 10,
        "generations": 5,
        "search_space": tpot_search_space,
        "n_jobs": num_cpus,
    }

    data_directory = "./Data/Raw_OpenML_Suite_271_Binary_Classification"
    split_directory = "./Data/Timing_Splits"
    base_save_folder = "./Results/TPOT_Comparison"
    taskids=[190412, 146818, 359955, 168757, 359956, 359958, 359962, 190137, 168911, 190392,
          189922, 359965, 359966, 359967, 190411, 146820, 359968, 359975, 359972, 168350,
          359973, 190410, 359971, 359988, 359989, 359979, 359980, 359992, 359982, 167120,
          359990, 189354, 360114, 359994]
    
    # Initialize Ray for cv_data_splitter to use ray.put()
    ray.init(num_cpus=num_cpus, ignore_reinit_error=True)
    
    num_reps = 20
    if machine == 'remote':
        tpot_loop_through_tasks(taskids, data_directory, split_directory, base_save_folder, num_reps, gp_params_remote)
    else:
        tpot_loop_through_tasks(taskids, data_directory, split_directory, base_save_folder, num_reps, gp_params_local)

    




    

