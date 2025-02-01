# GRSU

This is the official PyTorch implementation for the paper: **Search-Based Interaction For Conversation Recommendation via
Generative Reward Model Based Simulated User**

## Overview

- User preferences are often *multifaceted and complex*, we develop a simulated user, which can provide feedback to the items recommended by CRSs, enabling them to better capture intricate user preferences through *multi-turn interaction*.
- The simulated user is developed based on **generative reward models**
  - coarse-grained feedback: **generative item scoring**
  - fine-grained feedback: **attribute-based item critiquing**
  - these two actions are unified into an *instruction*-based format
- The interaction between CRSs and simulated users is formulated as **search**
  - We further propose an *efficient* candidate ranking method to improve the recommendation results derived from interaction

## Requirements

Please refer to `requirements.txt`.
Note that we only list the version of key packages here.

## Quick start

### Train the simulated user

First, download the generated instruction data from the link (available soon), and put them into the `dataset` directory.

Then, start training:

```bash
bash script/train/sft.sh [GPU_ID] [PORT]

e.g., bash script/train/sft.sh 0,1,2,3 8000
```

Training requires 4\*80G or 8\*40G gpus.

You can also directly download the model we trained from the link (available soon).

### Deploy the simulated user and CRS

We use vllm for serving and litellm for load balancing.

1. Serving

```bash
bash script/serve/vllm/user.sh [GPU_ID] [PORT]
bash script/serve/vllm/system.sh [GPU_ID] [PORT]

e.g., bash script/serve/vllm/user.sh 0 8000
```

Remember to set `model` in both scripts to the path of user/system.

For user, it should be the model you trained or downloaded from our link.
For system, it should be `microsoft/phi-4`. You can also try other models as the system.

2. Load balancing

We use [litellm](https://docs.litellm.ai/) for load balancing. You can refer to the doc for detailed usage.
Here, we use the python SDK of it.

In this step, you need to set up the config for using litellm.

We give the config template in `script/serve/litellm`.
In each config file, you need to fill in `[model]` and `[port]`.

- `[model]`: the value of `model` in script/serve/vllm/[user|system].sh
- `[port]`: the value used when serving the user/system model

If you want to use multiple nodes for load balancing, you need to modify `127.0.0.1` in `api_base` of the config to the corresponding IP address.

### Start interaction

```bash
python src/infer/main.py --config_file_path redial.yaml
python src/infer/main.py --config_file_path inspired.yaml
```

If the above command does not work, try `python -m src.infer.main`

The results are in the `result_file_dir`, which is defined in the yaml file.