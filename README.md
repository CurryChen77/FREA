# FREA: Feasibility-Guided Generation of Safety-Critical Scenarios with Reasonable Adversariality

<table style="border-collapse: collapse; width: 100%; table-layout: fixed;">
  <tr>
    <td style="border: 0; padding: 0; width: 50%;">
      <img src="./eval_analysis/figures/FREA-best1.gif" style="width: 99%; border: 2px solid gray;">
    </td>
    <td style="border: 0; padding: 0; width: 50%;">
      <img src="./eval_analysis/figures/FREA-best2.gif" style="width: 99%; border: 2px solid gray;">
    </td>
  </tr>
</table>

[![Static Badge](https://img.shields.io/badge/Arxiv-pdf-2be22f?logo=arxiv)](https://arxiv.org/abs/2406.02983)

<pre name="code" class="html">
<font color="red">[2024.6.05] <b>New: We release FREA paper on Arxiv. </b></font>
<font color="red">[2024.6.07] <b>New: The code is now released. </b></font></font>
</pre>

🌟FREA incorporates feasibility as guidance to generate adversarial yet AV-feasible, safety-critical scenarios for autonomous driving.🌟

<div style="text-align: center;">   <img style="border: 3px solid gray; width: 100%;" src="./eval_analysis/figures/FREA.jpg"/> </div>

## :page_with_curl: Outline

  - :art: [Setup](#Setup)
  - :books: [Usage](#Usage)
    - [Collect Off-line Data](#Collect-Off-line-Data)
    - [Train optimal feasible value function of AV](#Train-optimal-feasible-value-function-of-AV)
    - [Train adversarial policy of CBV](#Train-adversarial-policy-of-CBV)
    - [Evaluation](#Evaluation)
    - [Results Analysis](#Results-Analysis)
    - [Visualization](#Visualization)
  - :bookmark: [Citation](#citation)
  - :clipboard: [Acknowledgement](#Acknowledgement)

## :art: Setup

**Recommended system: Ubuntu 20.04 or 22.04**

Step 1: Install [Carla](https://carla.readthedocs.io/en/latest/start_quickstart/) (0.9.13 recommended)

Step 2: Setup conda enviroment

```bash
conda create -n frea python=3.8
conda activate frea
```

Step 3: Clone this git repo in an appropriate folder

```bash
git@github.com:CurryChen77/FREA.git
```

Step 4: Enter the repo root folder and install the packages:

```bash
cd FREA
pip install -r requirements.txt
pip install -e .
```

## :books: Usage

### Collect Off-line Data

```bash
# Launch CARLA
./CarlaUE4.sh -prefernvidia -windowed -carla-port=2000

# Launch in another terminal
python scripts/run.py --agent_cfg expert.yaml --scenario_cfg standard_train.yaml --mode collect_feasibility_data
```

### Train optimal feasible value function of AV

```bash
cd frea/feasibility/

# Train optimal feasible value function of AV
python train_feasibility.py
```

### Train adversarial policy of CBV

#### Train FREA

``````bash
# Launch CARLA
./CarlaUE4.sh -prefernvidia -windowed -carla-port=2000

# Train FREA
python scripts/run.py --agent_cfg expert.yaml --scenario_cfg fppo_adv_train.yaml --mode train_scenario
``````

#### Train FPPO-RS

```bash
# Launch CARLA
./CarlaUE4.sh -prefernvidia -windowed -carla-port=2000

# Train FPPO-RS
python scripts/run.py --agent_cfg expert.yaml --scenario_cfg fppo_rs_train.yaml --mode train_scenario
```

#### Train PPO

```bash
# Launch CARLA
./CarlaUE4.sh -prefernvidia -windowed -carla-port=2000

# Train PPO
python scripts/run.py --agent_cfg expert.yaml --scenario_cfg ppo_train.yaml --mode train_scenario
```

### Evaluation

#### Evaluation for data analysis (recording results)

```bash
# Launch CARLA
./CarlaUE4.sh -prefernvidia -windowed -carla-port=2000

# Evaluation FREA
python scripts/run.py --agent_cfg expert.yaml --scenario_cfg FPPO_adv_eval.yaml --mode eval --eval_mode analysis
```

#### Evaluation for video visualization

```bash
# Launch CARLA
./CarlaUE4.sh -prefernvidia -windowed -carla-port=2000

# Evaluation FREA
python scripts/run.py --agent_cfg expert.yaml --scenario_cfg FPPO_adv_eval.yaml --mode eval --eval_mode render
```

### Results Analysis

#### Results analysis of the paper

* [Evaluation Results](eval_analysis/plot_data/Eval_result.ipynb)
* [Learning Curve](eval_analysis/plot_data/Learning_Curve.ipynb)
* [Feasibility Results](frea/feasibility/feasibility_results.ipynb)

#### Generate your own results analysis

*Make sure the Evaluation has finished and the result are saved in [folder](./log/eval).*

```bash
# Process the recorded data
python eval_analysis/process_data/process_all_data.py

# Plot the evaluation result
python eval_analysis/plot_data/plot_evaluation_result.py
```

### Visualization

#### World spectator

```bash
# Launch CARLA
./CarlaUE4.sh -prefernvidia -windowed -carla-port=2000

# Set world spectator
python scripts/run.py --agent_cfg expert.yaml --scenario_cfg standard_eval.yaml --mode eval -sp
```

#### AV route

```bash
# Launch CARLA
./CarlaUE4.sh -prefernvidia -windowed -carla-port=2000

# Visualize AV route
python scripts/run.py --agent_cfg expert.yaml --scenario_cfg standard_eval.yaml --mode eval -viz_route
```

#### BEV map

 ```bash
 # Launch CARLA
 ./CarlaUE4.sh -prefernvidia -windowed -carla-port=2000
 
 # Visualize BEV map
 python scripts/run.py --agent_cfg expert.yaml --scenario_cfg FPPO_adv_eval.yaml --mode eval --eval_mode render
 ```

## :bookmark: Citation

If you find our paper useful, please kindly cite us via:
```
@article{chen2024frea,
  title={FREA: Feasibility-Guided Generation of Safety-Critical Scenarios with Reasonable Adversariality},
  author={Chen, Keyu and Lei, Yuheng and Cheng, Hao and Wu, Haoran and Sun, Wenchao and Zheng, Sifa},
  journal={arXiv preprint arXiv:2406.02983},
  year={2024}
}
```

## :clipboard: Acknowledgement

This implementation is based on code from several repositories. We sincerely thank the authors for their awesome work.
- [SafeBench](https://github.com/trust-ai/SafeBench)
- [FISOR](https://github.com/ZhengYinan-AIR/FISOR)
- [PlanT](https://github.com/autonomousvision/plant/tree/1bfb695910d816e70f53521aa263648072edea8e)
- [ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL/tree/master)
- [distance3d](https://github.com/AlexanderFabisch/distance3d)
- [King](https://github.com/autonomousvision/king/tree/main)
- [Two-Dimensional-Time-To-Collision](https://github.com/Yiru-Jiao/Two-Dimensional-Time-To-Collision)
