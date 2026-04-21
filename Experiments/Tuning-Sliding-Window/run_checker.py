import os

def check_slurm_outputs(directory, job_id=2156939, max_id=1080):
    missing_files = []
    incomplete_files = []

    for i in range(1, max_id + 1):
        filename = f"slurm-{job_id}_{i}.out"
        filepath = os.path.join(directory, filename)

        # Check if file exists
        if not os.path.exists(filepath):
            missing_files.append(i)
            continue

        # Check last line for "Archive saved to:"
        try:
            with open(filepath, "r") as f:
                lines = f.read().strip().splitlines()
                if not lines or not any("Archive saved to:" in line for line in lines):
                # if not lines or "Archive saved to:" not in lines[-1]:
                    incomplete_files.append(i)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            incomplete_files.append(i)

    # Print results
    print("Missing files:", ",".join(map(str, sorted(missing_files))))
    print("Incomplete files:", ",".join(map(str, sorted(incomplete_files))))


if __name__ == "__main__":
    # Replace with your directory path
    # directory = "/home/hernandezj45/Repos/ea-tpe"
    current_model = "DT"
    directory = f"/mnt/home/suzuekar/GECCO-2026-TPEC/Experiments/Tuning/{current_model}"
    
    # RF
    if current_model == "RF":
        # check_slurm_outputs(directory, 1351617, 146)
        # check_slurm_outputs(directory, 1351618, 146)
        # check_slurm_outputs(directory, 1351619, 146)
        # check_slurm_outputs(directory, 1351620, 146)
        check_slurm_outputs(directory, 2221119, 146)
        check_slurm_outputs(directory, 2221120, 146)
        check_slurm_outputs(directory, 2221121, 146)
        check_slurm_outputs(directory, 2221122, 146)

    # DT
    if current_model == "DT":
        # check_slurm_outputs(directory, 1379693, 146)
        # check_slurm_outputs(directory, 1379694, 146)
        # check_slurm_outputs(directory, 1379717, 146)
        # check_slurm_outputs(directory, 1379718, 146)
        # check_slurm_outputs(directory, 2083977, 146)
        # check_slurm_outputs(directory, 2083979, 146)
        # check_slurm_outputs(directory, 2083982, 146)
        # check_slurm_outputs(directory, 2083987, 146)
        check_slurm_outputs(directory, 2417843, 146)
        check_slurm_outputs(directory, 2417845, 146)
        check_slurm_outputs(directory, 2417847, 146)
        check_slurm_outputs(directory, 2417897, 146)

    # ET
    if current_model == "ET":
        # check_slurm_outputs(directory, 1385225, 146)
        # check_slurm_outputs(directory, 1385227, 146)
        # check_slurm_outputs(directory, 1385228, 146)
        # check_slurm_outputs(directory, 1385229, 146)
        # check_slurm_outputs(directory, 2086805, 146)
        # check_slurm_outputs(directory, 2086806, 146)
        # check_slurm_outputs(directory, 2086813, 146)
        # check_slurm_outputs(directory, 2086816, 146)
        check_slurm_outputs(directory, 2423399, 146)
        check_slurm_outputs(directory, 2423400, 146)
        check_slurm_outputs(directory, 2423401, 146)
        check_slurm_outputs(directory, 2423402, 146)

    # GB
    if current_model == "GB":
        # check_slurm_outputs(directory, 1391672, 146)
        # check_slurm_outputs(directory, 1391673, 146)
        # check_slurm_outputs(directory, 1391675, 146)
        # check_slurm_outputs(directory, 1391676, 146)
        check_slurm_outputs(directory, 2120311, 146)
        check_slurm_outputs(directory, 2120315, 146)
        check_slurm_outputs(directory, 2120320, 146)
        check_slurm_outputs(directory, 2120325, 146)

    # KSVC
    if current_model == "KSVC":
        # check_slurm_outputs(directory, 1395644, 146)
        # check_slurm_outputs(directory, 1395645, 146)
        # check_slurm_outputs(directory, 1395646, 146)
        # check_slurm_outputs(directory, 1395647, 146)
        check_slurm_outputs(directory, 2131611, 146)
        check_slurm_outputs(directory, 2131614, 146)
        check_slurm_outputs(directory, 2131617, 146)
        check_slurm_outputs(directory, 2131620, 146)

    # LSGD
    if current_model == "LSGD":
        # check_slurm_outputs(directory, 1399875, 146)
        # check_slurm_outputs(directory, 1399876, 146)
        # check_slurm_outputs(directory, 1399877, 146)
        # check_slurm_outputs(directory, 1399878, 146)
        check_slurm_outputs(directory, 2181780, 146)
        check_slurm_outputs(directory, 2181781, 146)
        check_slurm_outputs(directory, 2181782, 146)
        check_slurm_outputs(directory, 2181783, 146)

    # LSVC
    if current_model == "LSVC":
        # check_slurm_outputs(directory, 1404371, 146)
        # check_slurm_outputs(directory, 1404372, 146)
        # check_slurm_outputs(directory, 1404373, 146)
        # check_slurm_outputs(directory, 1404374, 146)
        check_slurm_outputs(directory, 2214446, 146)
        check_slurm_outputs(directory, 2214447, 146)
        check_slurm_outputs(directory, 2214448, 146)
        check_slurm_outputs(directory, 2214449, 146)
