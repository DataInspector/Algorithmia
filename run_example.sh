#!/usr/bin/env bash

set -o pipefail
set -ex

# pip is default package manager
pm=pip

current_dir="$(pwd)"
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

run_using_pip(){
    if [ ! -d env ]; then
        echo "Creating virtual environment..."
        if [ "${SCORING_PIPELINE_INSTALL_DEPENDENCIES}" == "0" ]; then
            virtualenv -p python3.6 --system-site-packages env
            env/bin/python -m pip install --upgrade --upgrade-strategy only-if-needed scoring_h2oai_experiment*.whl
        else
            virtualenv -p python3.6 env
            source env/bin/activate

            python -m pip install --upgrade --upgrade-strategy only-if-needed pip==19.1.1

            echo "Installing dependencies..."
            python -m pip install --upgrade --upgrade-strategy only-if-needed -r requirements.txt

            deactivate
        fi
    fi

    source env/bin/activate
    python example.py
    deactivate
}

run_using_conda(){
    env_name="$(head -n 1 environment.yml | cut -d ' ' -f 2)"
    create_conda_env ${env_name}
    source activate ${env_name}
    python example.py
    source deactivate
}

cd "${script_dir}"
source "./common-functions.sh"

main $@
cd "${current_dir}"
