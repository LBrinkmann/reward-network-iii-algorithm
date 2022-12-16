import papermill as pm
import yaml
import os
import subprocess


dev_param_dir = 'params/dev'
notebook_dir = 'notebooks'

def load_yaml(filename):
    with open(filename) as f:
        data = yaml.safe_load(f)
    return data


for file in os.listdir(notebook_dir):
    filename, ext = os.path.splitext(file)
    if ext == '.ipynb':
        parameter_file = os.path.join(dev_param_dir, f"{filename}.yml")
        notebook_file = os.path.join(notebook_dir, file)
        if os.path.exists(parameter_file):
            print(f'Update parameter in file {filename}')
            parameter = load_yaml(parameter_file)

            # add parameter to notebook
            pm.execute_notebook(notebook_file, notebook_file,
                                parameters=parameter, prepare_only=True)

            # create python file from jupyter notebook
            bashCommand = f"jupyter nbconvert --to script {notebook_file}"
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
        else:
            print(f'Skip file {filename}')
