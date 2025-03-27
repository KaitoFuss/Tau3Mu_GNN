
import os


config_base = """#!/bin/bash

#SBATCH --job-name=File_writing
#SBATCH --partition=short
#SBATCH --account=t3
#SBATCH --nodes=1                        # request to run job on single node                                       
#SBATCH --ntasks=8                     # request 10 CPU's (t3gpu01/02: balance between CPU and GPU : 5CPU/1GPU)   
#SBATCH --mem=8000M                     # memory (per job)
#SBATCH --time=00-01:00                   # time  in format DD-HH:MM
#SBATCH -o /work/ewiegers/logs/%x-%j.out  # Specify the output log file location
#SBATCH -e /work/ewiegers/logs/%x-%j.err  # Specify the error log file location
 
mkdir -p /scratch/$USER/${{SLURM_JOB_ID}}
export TMPDIR=/scratch/$USER/${{SLURM_JOB_ID}} 

"""




# Define QCD datasets with paths

input_folder = "/pnfs/psi.ch/cms/trivcat/store/user/ewiegers/Data/sChannel_mZprime-1500GeV_mDark-20GeV_rInv-0p5_alpha-peak/"
output_folder = "/pnfs/psi.ch/cms/trivcat/store/user/ewiegers/scores/Signal/sChannel_mZprime-1500GeV_mDark-20GeV_rInv-0p5_alpha-peak/"

alpha = 0

for file in os.listdir(input_folder):
    
    file_path = os.path.join(input_folder, file)
    
    if os.path.isfile(file_path) and file.endswith(".root"):
    
        job_name = f"job_{file.replace('.', '_')}"  # Replace dots in file name for safe job names
        config = config_base.format(job_name=job_name)
        
        provisional_folder = f"/scratch/$USER/${{SLURM_JOB_ID}}"
        edited_file = file.replace(".root", "_with_scores.root")
        edited_folder = f"root://t3dcachedb03.psi.ch/{output_folder}/"
        edited_output = f"{edited_folder}"
        
        print(f"Processing file: {file_path}")
        print(f"Provisional folder: {provisional_folder}")
        print(f"Edited file: {edited_file}")
        print(f"Edited output: {edited_output}")

    
        cmd = f"python score_per_file_example.py --file {file_path} --output_folder {provisional_folder} --alpha {alpha}"
        
        config += f"\n\n{cmd}\n"
        
        os.makedirs(output_folder, exist_ok=True)

        config += f"xrdcp {provisional_folder}/{edited_file} {edited_output}\n"

        print(f"xrdcp {provisional_folder}/{edited_file} {edited_output}")
        # Remove scratch directory at the end of the job
        config += f"rm -rf {provisional_folder}\n"

        # Write the job script to a file
        script_file = f"sub_files/{file.replace('.root', '')}_{output_folder.replace('/', '_')}.sh"
        os.makedirs("sub_files", exist_ok=True)
        with open(script_file, "w") as f:
            f.write(config)

        # Submit the job using sbatch
        os.system(f"sbatch {script_file}")