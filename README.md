# InfI-llm
## 📝 TODO List

 - [ ] Release demo video
 - [ ] Improve repository information
## Installation

The code has been tested with Python 3.10.8, CUDA 12.1.

### 1. Installing Dependencies
- We use adjusted versions of [habitat-sim](https://github.com/facebookresearch/habitat-sim) and [habitat-lab](https://github.com/facebookresearch/habitat-lab) as specified below.

- Install habitat-sim:
```
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim; git checkout tags/challenge-2022; 
pip install -r requirements.txt; 
python setup.py install --headless
```

- Install habitat-lab:
```
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab; git checkout tags/challenge-2022; 
pip install -e .
```

Back to the current repo, and replace the habitat folder in habitat-lab repo for the multi-robot setting: 

```
mv -r multi-robot-setting/habitat enter-your-path/habitat-lab
```

- Install [pytorch](https://pytorch.org/) according to your system configuration. The code is tested on torch v2.0.1, torchvision 0.15.2. 

- Install [detectron2](https://github.com/facebookresearch/detectron2/) according to your system configuration.

### 2. Download HM3D_v0.2 and MP3D datasets

#### Habitat Matterport
Download [HM3D_v0.2](https://aihabitat.org/datasets/hm3d/) and [MP3D](https://niessner.github.io/Matterport/) datasets using the download utility and [instructions](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md).

### 3. Download segmentation model

Download the [segmentation model](https://drive.google.com/file/d/1U0dS44DIPZ22nTjw0RfO431zV-lMPcvv/view?usp=share_link) in RedNet/model path.

### 4. Install YOLOv10

Follow the [README](detect/README.md) to install YOLOv10.

### 5. LLaMA-2 chat server

We now ship a self-contained FastAPI service under `llama_server/` that eagerly loads the
`meta-llama/Llama-2-7b-hf` checkpoint (config, tokenizer, weights, special tokens, and generation
defaults) as soon as it starts. The server binds to `0.0.0.0` so it can accept remote requests when
your firewall allows it.

1. Install the Python dependencies (PyTorch must match your CUDA setup):
   ```
   pip install -r requirements.txt
   ```
   > ℹ️ `torch` is not pinned. Install it manually following the guidance at
   > https://pytorch.org/get-started/locally/.

2. Export a Hugging Face token with access to Meta LLaMA weights if required:
   ```
   export HF_TOKEN=hf_xxx_your_token_here
   ```

3. Launch the service (defaults to `0.0.0.0:8000`):
   ```
   python -m llama_server --port 31511
   ```
   Override any setting via CLI flags (`--host`, `--model-id`, `--hf-token`) or through environment
   variables like `LLAMA_SERVER_PORT` and `LLAMA_SERVER_MODEL_ID`. To spin up multiple instances at once, add `--n-servers <count>`; each extra server binds to the next sequential port. For example:
   ```
   python -m llama_server --port 31511 --n-servers 3
   ```

4. Query the HTTP API from any machine that can reach the host/port:
   ```
   curl -X POST http://127.0.0.1:31511/v1/chat \
        -H "Content-Type: application/json" \
        -d '{"prompt": "Hello!"}'
   ```
   The `/v1/model_info` endpoint exposes the loaded special tokens and generation defaults.

5. Chat from a terminal without writing code:
   ```
   python -m llama_server.terminal_chat --host 127.0.0.1 --port 31511
   ```

The navigation scripts can connect to any reachable base URL via their `--base_url` flag.

## Setup
Install other requirements:
```
cd MCoCoNav/
pip install -r requirements.txt
```

### Setting up datasets
The code requires the datasets in a `data` folder in the following format (same as habitat-lab):
```
MCoCoNav/
  data/
    scene_datasets/
        hm3d_v0.2/
            val/
            hm3d_annotated_basis.scene_dataset_config.json
            hm3d_annotated_val_basis.scene_dataset_config.json
        mp3d/
    matterport_category_mappings.tsv
    object_norm_inv_perplexity.npy
    versioned_data
    objectgoal_hm3d_v2/
        train/
        val/
        val_mini/
```

## Evaluation
### Start the LLaMA-2 server:
```
python -m llama_server --port 31511
```

### Chat quickly from a terminal:
```
python -m llama_server.terminal_chat --host 127.0.0.1 --port 31511
```

### Eval 2-robot on HM3D_v0.2: 
```
python main.py -d ./VLM_EXP/multi_hm3d_2-robot/ \
  --num_agents 2 \
  --task_config tasks/multi_objectnav_hm3d.yaml \
  --base_url http://127.0.0.1:31511
```
