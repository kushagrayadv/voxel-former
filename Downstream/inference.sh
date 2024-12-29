instance_dir=$1
shift 1
# echo python inference.py --config-path $config_path hydra.job_logging.handlers.file.filename=infernece.log $@
python inference.py --config-dir ${instance_dir}/.hydra \
    'hydra.job_logging.handlers.file.filename=infernece.log' \
    "instance_dir=${instance_dir}" \
    'hydra.run.dir=${instance_dir}' \
    $@