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
pip install requirements.txt
```

## Running RL++ experiments

```bash
./run_experimental.sh

```
