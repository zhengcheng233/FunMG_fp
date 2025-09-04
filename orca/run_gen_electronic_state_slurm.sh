python /public/home/chengz/photomat/fs_users/multimodal__rela/FunMG_fp/orca/job_adv.py gen_electronic_slurm \
       --in_pth '/public/home/chengz/photomat/fs_projects/tii_stage2/qm_calc' \
       --slurm_task_pth '/public/home/chengz/photomat/fs_projects/tii_stage2/task' \
       --tmp_pth '/tmp/scratch/zhengcheng/FunMG_$SLURM_JOB_ID' \
       --orca_env 'module purge;source /public/home/chengz/apprepo/orca/6.0.1-openmpi416_gcc930/scripts/env.sh' \
       --platform_env '#SBATCH -p kshcnormal' \
       --script_pth '/public/home/chengz/photomat/fs_users/multimodal__rela/FunMG_fp/orca/job_adv.py' \
       --nproc 30 \
       --out_pth "/public/home/chengz/photomat/fs_projects/tii_stage2/qm_calc"  
