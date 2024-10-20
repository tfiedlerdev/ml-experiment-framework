# Machine Learning Experiment Framwork
This framework attempts to streamline the machine learning research process to enable
- zero effort reproducability
- a single source of truth for the definition of which arguments are allowed
- IntelliSense for available arguments for a certain experiment
- code reusability
- collaboration through a proper software architecture instead of copy paste experimenting
- zero effort logging and history plotting

The assumptions to achieve this are that all experiment share some basic logic. 
They all have a 
- ML model (here a pytorch Module with an additional interface)
- Dataset
- train loop with a specified number of epochs
- an optimizer
- a scheduler
- training should be tracked via WandB if enabled via `--use_wandb=true`

The idea is that all of these previously listed concepts follow an extendable base interface for the corresponding concept so that the framework can work with them (e.g. BaseModel, BaseDataset, ..).
Each of these Modules also has their own Pydantic Model, which specifies its variables. In the specific experiment using the modules, the Pydantic Models are sticked together and a Vanilla Python ArgParser is constructed automatically from the Pydantic model.
For arguments that do not change across experiments such as API keys etc., a yaml config is used.
## Getting started
1. Fork this repository
2. Create conda environment from `environment.yaml`: `conda env create --file environment.yaml`
3. Run MNIST experiment via `python run.py --experiment_type=mnist --use_cuda=false --hidden_sizes="[64]"`, you'll be prompted to fill out a config YAML to specify directories for cache files and the experiment results
4. Run the command again, training should run and you should see the experiment results in the specified folder


