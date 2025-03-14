# UW's CSE517 - Reproduction of a paper final project

We attempt to reproduce the [StablePrompt paper](https://arxiv.org/pdf/2410.07652) from EMNLP 2024. Official Code Implementation of StablePrompt can be found [here](https://github.com/kmc0207/Stableprompt). We run our experiments on top of that base code.


## Setting Up the Environment

In the original paper, this is how the environment was setup:

```bash
docker run pytorch:latest
pip install -r requirements.txt
git clone https://github.com/keirp/automatic_prompt_engineer.git
```

However, libraries were not versioned and that does not work anymore. So we construct a fully versioned requirements file. This is how our reproduction environment is setup:

```bash
conda create --name stableprompt python=3.10.12
pip install requirements_repr.txt
```

## Running experiments

The original authors have provided handy bash scripts that execute each task's code. We build slurm files on top of those to execute them in a multi-gpu setting.

### Experiment 4.1: Few-shot Text Classification
```bash
./run_fewshot.sh
# OR for multi-gpu on slurm
sbatch run_fewshot.slurm
```

### Experiment 4.2: Induction Task
```bash
./run_BBII.sh
./run_II.sh
# OR for multi-gpu on slurm
sbatch run_BBII.slurm
sbatch run_II.slurm
```

### Experiment 4.3: Question Answering
```bash
./run_QA.sh
# OR for multi-gpu on slurm
sbatch run_QA.slurm
```

### Additional Experiment

To run our proposed RL++ method, you need to setup another environment, as GRPO required newer versions of the libraries.

```bash
# todo
```
