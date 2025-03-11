import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import (
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
    GRPOConfig,
    GRPOTrainer,
)
from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)
from accelerate import Accelerator
import argparse
import numpy as np
import wandb
import copy
import random
import heapq
import utils
from dataset_utils import (
    load_all_dataset,
    dataset_dicts,
    load_qa_dataset,
    qa_dicts,
    load_generation_dataset,
)
from peft import LoraConfig
from peft import get_peft_model
from datasets import Dataset
import random
import deepspeed
import json

from accelerate.utils.other import is_compiled_module
from contextlib import contextmanager
from accelerate.utils import (
    broadcast_object_list,
    gather,
    gather_object,
    is_peft_model,
    set_seed,
)
from packaging import version
import itertools


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",  # "google/gemma-7b-it",  # "meta-llama/Llama-3.2-1B-Instruct"
    )
    parser.add_argument(
        "--agent_model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",  # "google/gemma-7b-it",  # "meta-llama/Llama-3.2-1B-Instruct"
    )
    parser.add_argument("--task", type=str, default="classification")
    parser.add_argument("--dataset", type=str, default="sst2")
    parser.add_argument("--verbalizer", type=str, nargs="+", default=None)
    parser.add_argument(
        "--cache_dir", type=str, default="gscratch/ark/graf/grpo_cache/llm/"
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_prompt_length", type=int, default=100)
    parser.add_argument("--train_data_per_labels", type=int, default=16)
    parser.add_argument("--num_example", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=256)
    parser.add_argument(
        "--meta_prompt",
        type=str,
        default="""I gave a friend an instruction and five inputs. 
                        The friend read the instruction and wrote an output for every one of the inputs.
                        Here are the input-output pairs: \n
                        """,
    )
    parser.add_argument("--prompt_per_example", type=int, default=4)
    parser.add_argument("--update_term", type=int, default=15)
    parser.add_argument("--update_threshold", type=float, default=0.05)
    parser.add_argument("--num_test_example", type=int, default=20)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()
    return args


def sample_random_elements(dataset, n):
    """
    Sample n random elements from a dataset.

    Args:
        dataset: The dataset to sample from
        n: Number of elements to sample

    Returns:
        A subset of the dataset containing n random elements
    """

    # Make sure we don't try to sample more elements than available
    n = min(n, len(dataset))

    # Get random indices without replacement
    indices = random.sample(range(len(dataset)), n)

    # Create a subset based on the selected indices
    sampled_dataset = [dataset[i] for i in indices]

    return sampled_dataset


def add_hooks(model) -> None:
    """Adds the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    if not hasattr(
        model, "optimizer"
    ):  # before the first training step, the model has no optimizer
        return
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer
    else:
        raise RuntimeError("The model optimizer is None, which is not yet supported.")
    if version.parse(deepspeed.__version__) >= version.parse("0.16.4"):
        # Account for renaming in https://github.com/deepspeedai/DeepSpeed/pull/6847
        optimizer_offload._register_deepspeed_module(optimizer_offload.module)
    else:
        optimizer_offload._register_hooks_recursively(optimizer_offload.module)


def remove_hooks(model) -> None:
    """Removes the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    if not hasattr(
        model, "optimizer"
    ):  # before the first training step, the model has no optimizer
        return
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer
    else:
        raise RuntimeError("The model optimizer is None, which is not yet supported.")

    for param in iter_params(optimizer_offload.module, recurse=True):
        param.ds_active_sub_modules.clear()

    for hook in optimizer_offload.forward_hooks:
        hook.remove()
    for hook in optimizer_offload.backward_hooks:
        hook.remove()

    optimizer_offload.forward_hooks = []
    optimizer_offload.backward_hooks = []


def get_all_parameters(sub_module, recurse=False):
    return itertools.chain(
        sub_module.named_parameters(recurse=recurse),
        sub_module.ds_external_parameters(),
    )


def iter_params(module, recurse=False):
    return [param for _, param in get_all_parameters(module, recurse)]


@contextmanager
def unwrap_model_for_generation(
    model,
    accelerator: "Accelerator",
    gather_deepspeed3_params: bool = True,
):
    """
    Context manager to unwrap distributed or accelerated models for generation tasks.

    Args:
        model (`Union[DistributedDataParallel, DeepSpeedEngine]`):
            Model to be unwrapped.
        accelerator (`~accelerate.Accelerator`):
            Accelerator instance managing the model.
        gather_deepspeed3_params (`bool`, *optional*, defaults to `True`):
            Whether to gather weights for DeepSpeed ZeRO Stage 3 models. If `False`, skips parameter gathering, which
            can be more memory-efficient but may lead to slower generation times.

    Yields:
        Unwrapped model.

    Example:
    ```python
    with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
        generated_outputs = unwrapped_model.generate(input_ids)
    ```
    """
    unwrapped_model = accelerator.unwrap_model(model)
    if (
        accelerator.state.deepspeed_plugin is not None
        and accelerator.state.deepspeed_plugin.zero_stage == 3
    ):
        if not gather_deepspeed3_params:
            yield accelerator.unwrap_model(model)
        else:
            with deepspeed.zero.GatheredParameters(model.parameters()):
                remove_hooks(model)
                yield accelerator.unwrap_model(model)
                add_hooks(model)
    else:
        yield unwrapped_model


def _move_model_to_vllm(trainer, model):
    with unwrap_model_for_generation(
        model,
        trainer.accelerator,
        gather_deepspeed3_params=trainer.args.ds3_gather_for_generation,
    ) as unwrapped_model:
        if is_compiled_module(unwrapped_model):
            unwrapped_model = unwrapped_model._orig_mod
        if is_peft_model(unwrapped_model):
            unwrapped_model.merge_adapter()
            state_dict = unwrapped_model.state_dict()
            # Remove base_model and base_layer prefixes
            state_dict = {
                k.removeprefix("base_model.model.").replace(".base_layer", ""): v
                for k, v in state_dict.items()
            }
            # Remove values with adapter prefix (example: "_lora")
            state_dict = {
                k: v for k, v in state_dict.items() if unwrapped_model.prefix not in k
            }
            # When module to save, remove its prefix and discard the original module
            state_dict = {
                k.replace("modules_to_save.default.", ""): v
                for k, v in state_dict.items()
                if "original_module" not in k
            }
        else:
            state_dict = unwrapped_model.state_dict()
        if trainer.accelerator.is_main_process:
            llm_model = (
                trainer.llm.llm_engine.model_executor.driver_worker.model_runner.model
            )
            llm_model.load_weights(state_dict.items())
        # Unmerge the adapter to restore the model to its original state.
        # This must be done after loading weights to ensure they correspond to the merged state.
        if is_peft_model(unwrapped_model):
            unwrapped_model.unmerge_adapter()


class PromptDataset(Dataset):
    def __init__(
        self,
        validation_dataset,
        verbalizer,
        num_example,
        meta_prompt,
        agent_tokenizer,
        steps=1024,
    ):
        self.validation_dataset = validation_dataset
        self.verbalizer = verbalizer
        self.num_example = num_example
        self.meta_prompt = meta_prompt
        self.agent_tokenizer = agent_tokenizer

        # Generate the examples once during initialization
        self.examples = utils.got_example(
            validation_dataset, verbalizer, shot=num_example
        )

        # Create the prompts for each item
        self.prompts = [{**self._generate_prompts(), "idx": i} for i in range(steps)]

    def _generate_prompts(self):
        # Generate the prompt text for each item
        query_text = [
            {"role": "user", "content": self.meta_prompt + "\n" + self.examples},
            {"role": "assistant", "content": "The Instruction is : "},
        ]

        prompt_text = self.agent_tokenizer.apply_chat_template(
            query_text, tokenize=False, continue_final_message=True
        )

        return {"prompt": prompt_text}

    def __len__(self):
        # Return the number of prompts
        return len(self.prompts)

    def __getitem__(self, idx):
        # Return the prompt at the given index
        if isinstance(idx, (list, tuple)):
            return self.__getitems__(idx)
        return self.prompts[idx]

    def __getitems__(self, indices):
        # Support batch retrieval
        return [self.prompts[idx] for idx in indices]


def main():

    args = parser_args()
    # device = "cuda:0"

    if args.local_rank == 0:
        wandb.init(
            project="algprompt_" + args.task + "_" + args.dataset,
            config=args,
            name=args.task
            + "_"
            + args.dataset
            + "_"
            + args.agent_model
            + "_"
            + args.target_model,
            # mode="disabled",
        )

    if args.task == "classification":
        dataset = load_all_dataset(args.dataset)
        train_dataset = dataset[0]
        test_dataset = dataset[2]
        # test_dataset = utils.create_balanced_subset(test_dataset,100)
        if args.verbalizer is None:
            verbalizer = dataset_dicts(args.dataset)
        num_labels = len(verbalizer)
        train_dataset, validation_dataset = utils.create_balanced_subset_and_validation(
            train_dataset,
            args.train_data_per_labels * num_labels,
        )

    elif args.task == "qa":
        dataset = load_qa_dataset(args.dataset)
        train_dataset = dataset[0]
        test_dataset = dataset[2]
        test_dataset = utils.create_balanced_subset(test_dataset, 100)
        if args.verbalizer is None:
            verbalizer = qa_dicts()
        num_labels = len(verbalizer)
        validation_dataset = train_dataset

    elif args.task == "generation":
        dataset = load_generation_dataset(args.dataset)
        train_dataset = dataset[0]
        test_dataset = dataset[2]
        test_dataset = utils.create_balanced_subset(test_dataset, 100)
        verbalizer = None
        validation_dataset = train_dataset

    # make dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )

    # print("train_data_size", len(train_dataset))
    # print("test_data_size", len(test_dataset))
    # load agent model
    """config = PPOConfig(
        model_name=args.agent_model,
        learning_rate=1e-5,
        batch_size=args.prompt_per_example,
        mini_batch_size=args.prompt_per_example,
        log_with="wandb",
    )"""

    # grpo_config = GRPOConfig(
    # model_name=args.agent_model,
    #    learning_rate=1e-5,
    # batch_size=args.prompt_per_example,
    # num_gen
    # log_with="wandb",
    #   use_vllm=False,  # Set to True if you have vLLM installed for speed
    # GRPO specific parameters
    #  beta=0.04,  # KL coefficient
    # num_iterations=1,
    # epsilon=0.2,  # Clipping parameter
    # Reference model synchronization
    # sync_ref_model=True,
    # ref_model_mixup_alpha=0.6,
    # ref_model_sync_steps=args.update_term,  # Use the same update frequency as in PPO
    # )

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    agent_tokenizer = AutoTokenizer.from_pretrained(
        args.agent_model, cache_dir=args.cache_dir
    )
    agent_model = AutoModelForCausalLM.from_pretrained(
        args.agent_model,
        # torch_dtype=torch.bfloat16,
        # device_map={"": Accelerator().local_process_index},
        # peft_config=lora_config,
        cache_dir=args.cache_dir,
        # use_cache=False,
    )

    """ref_model = AutoModelForCausalLM.from_pretrained(
        args.agent_model,
        # torch_dtype=torch.bfloat16,
        # device_map={"": Accelerator().local_process_index},
        # peft_config=lora_config,
        cache_dir=args.cache_dir,
    )"""
    # ref_model = get_peft_model(ref_model, lora_config)
    # agent_model = get_peft_model(agent_model, lora_config)
    agent_tokenizer.pad_token = agent_tokenizer.eos_token
    agent_tokenizer.pad_token_id = agent_tokenizer.eos_token_id
    device = torch.device(f"cuda:{args.local_rank}")
    # load target model

    # target_tokenizer = AutoTokenizer.from_pretrained(
    #    args.target_model, cache_dir=args.cache_dir
    # )
    # target_model = AutoModelForCausalLM.from_pretrained(
    #    args.target_model,
    #    cache_dir=args.cache_dir,
    #    torch_dtype=torch.bfloat16,
    # device_map=torch.device(f"cuda:{args.local_rank}"),  # args.local_rank
    # ).to(device)
    # target_model.config.pad_token_id = target_tokenizer.eos_token_id
    # target_tokenizer.pad_token = target_tokenizer.eos_token

    target_tokenizer = agent_tokenizer
    prompt_dataset = PromptDataset(
        validation_dataset,
        verbalizer,
        args.num_example,
        args.meta_prompt,
        agent_tokenizer,
        steps=args.epochs,
    )

    def prompt_reward_function(
        prompts,
        completions,
        task_dataset=train_dataset,
        batch_size=args.batch_size,
        soft=True,
        **kwargs,
    ):
        # Get evaluation scores for the prompts

        """
        [inputs,labels] <- random batch from train dataset dataloader
        new_dict ={
                'text' : inputs,
                'label' : labels
            }
            new_ds = Dataset.from_dict(new_dict)
        """

        subset = sample_random_elements(task_dataset, batch_size)

        new_dict = {
            "text": [s["text"] for s in subset],
            "label": [s["label"] for s in subset],
        }
        new_ds = Dataset.from_dict(new_dict)

        # _move_model_to_vllm(trainer, target_model)

        with torch.no_grad():

            accuracies, softmax_diff = utils.evaluation_sdX(
                completions,
                new_ds,
                trainer.ref_model,  ## trainer.ref_model
                target_tokenizer,
                device,
                verbalizer.values(),
                batch_size=4,
            )

            if soft:

                # Calculate rewards (combining accuracy and softmax difference)
                rewards = [
                    0.01 * softmax_diff[i].item() + 30 * accuracies[i].item()
                    for i in range(len(completions))
                ]
            else:
                rewards = [accuracies[i].item() * 30 for i in range(len(completions))]

        # print(rewards)
        # import pdb

        # pdb.set_trace()

        return rewards

    def get_last_generation():
        df_path = wandb.run.summary["completions"]["path"]
        with open("wandb/latest-run/files/" + df_path, "r") as f:
            df = json.load(f)

        df = dict(zip(df["columns"], df["data"][0]))

        return df

    def perform_eval(prompt, completion, dataset, **kwargs):

        prompts = [prompt]
        completions = [completion]

        rewards = prompt_reward_function(
            prompts,
            completions,
            task_dataset=dataset,
            batch_size=len(dataset),
            soft=False,
            **kwargs,
        )

        metrics = {"acc": torch.mean(torch.tensor(rewards) / 30).item()}

        return metrics

    class ValidationAccuracyCallback(TrainerCallback):
        """
        Callback to evaluate model on validation dataset during training with TRL.
        Computes and logs validation accuracy after each evaluation step.
        """

        def __init__(
            self,
            validation_dataset: Dataset,
            eval_steps=2,
        ):
            """
            Args:
                validation_dataset: The validation dataset to evaluate on
                tokenizer: The tokenizer used for the model
                compute_metrics_fn: Optional custom metrics function
                eval_batch_size: Batch size to use during evaluation
            """
            self.validation_dataset = validation_dataset
            self.best_reward = 0.0
            self.eval_steps = eval_steps
            self.best_completion = None
            self.best_prompt = None

        def on_step_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):

            if (
                state.global_step % self.eval_steps == 0
                and len(state.log_history) > 0
                and args.local_rank == 0
            ):
                print(f"Step {state.global_step} - Logging validation metrics...")

                df = get_last_generation()

                metrics = perform_eval(
                    df["prompt"], df["completion"], self.validation_dataset, **kwargs
                )

                metrics = {"val_acc": metrics["acc"]}

                state.log_history[-1]["val_acc"] = metrics["val_acc"]

                wandb.log(metrics)

                # Track best accuracy
                if metrics.get("val_acc", 0) > self.best_reward:
                    self.best_reward = metrics.get("val_acc", 0)
                    print(
                        f"\n*** New best validation accuracy: {self.best_reward:.4f} ***\n"
                    )
                    self.best_completion = df["completion"]
                    self.best_prompt = df["prompt"]

            return control

    validation_callback = ValidationAccuracyCallback(validation_dataset)

    training_args = GRPOConfig(
        output_dir=f"{args.agent_model}-GRPO",
        learning_rate=5e-6,
        bf16=True,
        bf16_full_eval=True,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=1e-6,
        warmup_steps=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        # train_batch_size=16,
        gradient_checkpointing=True,
        beta=1e-5,
        lr_scheduler_type="cosine_with_restarts",
        logging_steps=1,
        num_generations=args.num_generations,  # 4,
        max_prompt_length=512,
        max_completion_length=args.max_prompt_length,
        num_train_epochs=1,
        save_steps=1024,
        report_to=["wandb"],
        deepspeed="ds_config_zero3.json",
        local_rank=args.local_rank,
        log_completions=True,
        log_on_each_node=False,
        use_vllm=True,
        vllm_device="cuda:0",
        vllm_gpu_memory_utilization=0.3,
        vllm_dtype=torch.bfloat16,
        vllm_max_model_len=1024,
        # eval_steps=1,
        # eval_strategy="steps",
        # ref_model_mixup_alpha=0.95,
        # sync_ref_model=True,
        # ref_model_sync_steps=args.update_term,
    )

    trainer = GRPOTrainer(
        model=agent_model,
        # tokenizer=agent_tokenizer,
        reward_funcs=prompt_reward_function,  # Our custom reward function
        args=training_args,
        processing_class=agent_tokenizer,
        # peft_config=lora_config,
        train_dataset=prompt_dataset,
        callbacks=[validation_callback],
    )

    trainer.train()

    if args.local_rank == 0:

        metrics = perform_eval(
            validation_callback.best_prompt,
            validation_callback.best_completion,
            test_dataset,
        )

        print("Final test accuracy:", metrics["acc"])


if __name__ == "__main__":
    main()
