import numpy as np
import tpot
import sklearn
from sklearn.metrics import accuracy_score
from Source.tpot_comparison.param_space_conversion import generate_tpot_search_space
from functools import partial
from Source.Base.data_utils import load_data, preprocess_train_test
from typing import List
import os
import time
import random
import pickle
import traceback
import argparse
from Source.tpot_comparison.prepare_data_for_tpot import get_splits

def custom_objective_function(estimator, X_train_splits, X_val_splits, y_train_splits, y_val_splits):
    assert len(X_train_splits) == 5, f"Expected 5 train splits, got {len(X_train_splits)}"
    assert len(X_val_splits) == 5, f"Expected 5 val splits, got {len(X_val_splits)}"
    scores = []
    for index in range(5):
        X_train, X_val = X_train_splits[index], X_val_splits[index]
        y_train, y_val = y_train_splits[index], y_val_splits[index]
        assert len(X_train) > 0, f"Empty training set at fold {index}"
        assert len(X_val) > 0, f"Empty validation set at fold {index}"

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
                assert os.path.exists(rep_dir), f"Replicate directory does not exist: {rep_dir}"
                X_train, X_test, y_train, y_test = load_data(taskid, data_directory, rep_dir)
                assert len(X_train) > 0, f"Empty training data for task {taskid}, rep {r}"
                assert len(X_test) > 0, f"Empty test data for task {taskid}, rep {r}"
                assert X_train.shape[1] == X_test.shape[1], f"Train/test feature mismatch: {X_train.shape[1]} vs {X_test.shape[1]}"
                # Get CV splits for custom objective
                X_train_splits, X_val_splits, y_train_splits, y_val_splits = get_splits(taskid, r, data_directory, split_directory)
                assert len(X_train_splits) == 5, f"Expected 5 CV splits, got {len(X_train_splits)}"
                
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
                                            classification = True,
                                            max_eval_time_mins = 1440, # 24 hours per evaluation  
                                            max_time_mins = None,
                                            n_jobs=ga_params['n_jobs'],
                                            verbose=5,
                                            random_state=super_seed,
                                            )

                          # fit best individual on full training data and evaluate on test set
                    X_train_transformed, y_train, X_test_transformed, y_test = preprocess_train_test(X_train,
                                                                                         y_train,
                                                                                         X_test,
                                                                                         y_test,
                                                                                         taskid,
                                                                                         data_directory)
                    assert len(X_train_transformed) > 0, f"Empty training data for task {taskid}, rep {r}"
                    assert len(X_test_transformed) > 0, f"Empty test data for task {taskid}, rep {r}"
                    assert X_train_transformed.shape[1] == X_test_transformed.shape[1], f"Train/test feature mismatch: {X_train.shape[1]} vs {X_test.shape[1]}"

                    linear_est.fit(X_train_transformed, y_train)
                    print("Ending the fitting process. ")
                    assert hasattr(linear_est, 'fitted_pipeline_'), "TPOT fitting failed - no fitted_pipeline_ attribute"
                    best_pipeline = linear_est.fitted_pipeline_
                    assert best_pipeline is not None, "Fitted pipeline is None"
                    
                    accuracy_scorer = sklearn.metrics.get_scorer("accuracy")
                    
                    # If y in not in [0,1,...N], TPOT encodes labels for classification tasks; using the same encoding here
                    if linear_est.label_encoder_ is not None:
                        y_train = linear_est.label_encoder_.fit_transform(y_train)
                        y_test = linear_est.label_encoder_.transform(y_test)

                    train_accuracy = accuracy_scorer(best_pipeline, X_train_transformed, y_train)
                    train_score = {"train_accuracy": train_accuracy}
                    test_accuracy = accuracy_scorer(best_pipeline, X_test_transformed, y_test)
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

                    # Also save the distribution of classifers in all evaluated individuals
                    all_eval_inds = linear_est.evaluated_individuals

                    # all_eval_inds is a pandas dataframe with 'Instance' column storing sklearn pipelines
                    
                    # Extract classifier distribution from evaluated individuals
                    classifier_counts = {}
                    for _, row in all_eval_inds.iterrows():
                        pipeline = row['Instance'] # Pipeline is just a single classifier here
                        classifier_name = pipeline.__class__.__name__  # Get the classifier name
                        if classifier_name in classifier_counts:
                            classifier_counts[classifier_name] += 1
                        else:
                            classifier_counts[classifier_name] = 1

                    this_score["classifier_distribution"] = classifier_counts
                    this_score["num_evaluated_individuals"] = len(all_eval_inds)

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
    parser.add_argument('--num_reps', type=int, default=21, help='Number of Replicates')
    args = parser.parse_args()
    num_cpus = args.num_cpus
    machine = args.machine
    num_reps = args.num_reps

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
    base_save_folder = "Results/tpot"
    taskids = [190412, 146818, 359955, 168757, 359956, 359958, 359962, 190137, 168911, 359965, 190411, 146820, 359968, 359975, 359972, 168350, 359971]
    assert os.path.exists(data_directory), f"Data directory does not exist: {data_directory}"
    assert os.path.exists(split_directory), f"Split directory does not exist: {split_directory}"
    assert num_cpus > 0, f"Invalid num_cpus: {num_cpus}"
    assert num_reps > 0, f"Invalid num_reps: {num_reps}"
    print(f"Starting TPOT experiments with {num_cpus} CPUs, {num_reps} replicates on {machine}")
    
    if machine == 'remote':
        tpot_loop_through_tasks(taskids, data_directory, split_directory, base_save_folder, num_reps, gp_params_remote)
    else:
        tpot_loop_through_tasks(taskids, data_directory, split_directory, base_save_folder, num_reps, gp_params_local)
