# RESA: Reasonable Evaluation via Safety Assurance

# Content

* [Eval](#Eval)

* [Train Agent](#Train-Agent)
* [Train Scenario](#Train-Scenario)
* [Collect_feasibility_data](#Collect-feasibility-data)
* [Visualization](#Visualization)

## Eval

### Desktop Users

```bash
# Launch CARLA
./CarlaUE4.sh -prefernvidia -windowed -carla-port=2000

# Launch in another terminal
python scripts/run.py --agent_cfg expert.yaml --scenario_cfg standard_eval.yaml --mode eval -sp

# Use feasibility to help Eval
python scripts/run.py --agent_cfg expert.yaml --scenario_cfg standard_eval.yaml --mode eval -sp -fe 

```

### Remote Users

#### Remote training

```bash
# Launch CARLA with headless mode
./CarlaUE4.sh -prefernvidia -RenderOffScreen -carla-port=2000

# Another terminal no display pygame
python scripts/run.py --agent_cfg expert.yaml --scenario_cfg ppo_train.yaml --mode train_scenario

# Showing the memory usage
mprof run --python scripts/run.py --agent_cfg expert.yaml --scenario_cfg ppo_train.yaml --mode train_scenario
```

#### Visualize pygame window

```bash
# Launch CARLA with headless mode
./CarlaUE4.sh -prefernvidia -RenderOffScreen -carla-port=2000

# Another terminal, Launch on the virtual display
DISPLAY=:10 python scripts/run.py --agent_cfg expert.yaml --scenario_cfg ppo_eval.yaml --mode eval --render
```

* local open terminal

```bash
ssh -X username@host
```

## Train Agent

### Usage

``````bash
# Launch CARLA
./CarlaUE4.sh -prefernvidia -windowed -carla-port=2000

# Launch in another terminal
python scripts/run.py --agent_cfg ppo.yaml --scenario_cfg standard_train.yaml --mode train_agent

# profile the memory usage
mprof run python scripts/run.py --agent_cfg ppo.yaml --scenario_cfg standard_train.yaml --mode train_agent
``````


### Policy

* **behavior:** Carla default agent (no state)

* **expert:** Carla leader board default rule-based agent (ego state)
* **plant:**  transformer based planning agent, output ego's future waypoints (ego state)
* **PPO:** RL-based agent

### state/observation

For learnable agent (PPO......)

* **Ego obs**
  
  |           vehicle            |  x   |  y   | extent_x | extent_y | yaw  | velocity |
  | :--------------------------: | :--: | :--: | :------: | :------: | ---- | :------: |
  | ego state (relative to ego)  |      |      |          |          |      |          |
  | BV_1 state (relative to ego) |      |      |          |          |      |          |
  | BV_2 state (relative to ego) |      |      |          |          |      |          |

### Action

2-dim continues action

* throttle
* steering
* brake

## Train Scenario

### Usage

```bash
# Launch CARLA
./CarlaUE4.sh -prefernvidia -windowed -carla-port=2000

# Launch in another terminal
python scripts/run.py --agent_cfg expert.yaml --scenario_cfg ppo_train.yaml --mode train_scenario
```

### Policy

1. PPO
2. **FPPO (feasibility guided PPO)**
3. standard (autopilot)

### optional

* Select the CBV method **based on the CBV candidate**

  1. **attention-based** (default, attention weight)
  2. **rule-based** (nearest)

```bash
python scripts/run.py --agent_cfg expert.yaml --scenario_cfg standard_eval.yaml --mode train_scenario  --CBV_selection 'attention-based'  # different method of selecting controlled bv
```

* Input state

  **Actor info**

  |           vehicle            |  x   |  y   | extent_x | extent_y | yaw  | velocity |
  | :--------------------------: | :--: | :--: | :------: | :------: | :--: | -------- |
  | CBV state (relative to CBV)  |      |      |          |          |      |          |
  | ego state (relative to CBV)  |      |      |          |          |      |          |
  | BV_1 state (relative to CBV) |      |      |          |          |      |          |
  
  **Tips: if no BV then the corresponding state will set to 0**
  
* Output action

  2-dim continues action

  acc ~ [-3.0, 3.0]

  steer ~ [-0.3, 0.3]

## Collect feasibility data

### Usage

```bash
# Launch CARLA
./CarlaUE4.sh -prefernvidia -windowed -carla-port=2000

# Launch in another terminal
python scripts/run.py --agent_cfg expert.yaml --scenario_cfg ppo_train.yaml --mode collect_feasibility_data
```

### Policy

* HJ-Reachability

### Input state

[yaml file](safebench/feasibility/config/HJR.yaml)  ```obs_type: "ego_info" (default)```

## Visualization

### World spectator

```bash
# Launch CARLA
./CarlaUE4.sh -prefernvidia -windowed -carla-port=2000

# Launch in another terminal
python scripts/run.py --agent_cfg behavior.yaml --scenario_cfg standard_eval.yaml --mode eval -sp  # Set world spectator on the first scenario's ego
```

### Ego route

```bash
# Launch CARLA
./CarlaUE4.sh -prefernvidia -windowed -carla-port=2000

# Launch in another terminal
python scripts/run.py --agent_cfg behavior.yaml --scenario_cfg standard_eval.yaml --mode eval --viz_route  # visualization the global route
```

### State encoder attn map

[state encoder yaml](safebench/agent/config/state_encoder.yaml)   ```viz_attn_map=True (default)```

### render the whole pygame window

 ```bash
 # Launch CARLA
 ./CarlaUE4.sh -prefernvidia -windowed -carla-port=2000
 
 # Launch in another terminal
 python scripts/run.py --agent_cfg behavior.yaml --scenario_cfg standard_eval.yaml --mode eval --render  # show the pygame window
 ```

### enable the sem camera for visualization

```bash
# Launch CARLA
./CarlaUE4.sh -prefernvidia -windowed -carla-port=2000

# Launch in another terminal
python scripts/run.py --agent_cfg behavior.yaml --scenario_cfg standard_eval.yaml --mode eval --render --envable_sem # show the pygame window and enable the 3rd-person view using the semantic segmentation camera
```

## Metric

### statistical result:

The evaluation result are the statistical result up to now

* collision rate
* **unavoidable collision**
* **near rate**
* **near miss rate**
* out of road length
* distance to route
* incomplete route
* running time
* **final score**

## Tricks

### CBV selection candidate

* Except the ego vehicle
* Except the BVs which are in the opposite lane side, but on the same road
* Except the BVs behind the ego vehicle and the abs yaw angle difference > 45 degree

### Theory

**Feasible region:**

$$\mathop{\max}\limits_{\pi} \mathbb{E}_s [A^{\pi}_r(s) \cdot \mathbb{1}_{s \in \it{S}^*_f}]$$

**Infeasible region**

$$\mathop{\max}\limits_{\pi} \mathbb{E}_s [-A^{*}_h(s) \cdot \mathbb{1}_{s \notin \it{S}^*_f}]$$

**Definition of Feasible region:**

$$\mathit{S}^*_{f} :=\{s\mid V^*_{h}(s) \leq 0 \}$$

$$V^*_h (s)\leq 0 \Rightarrow \mathop{\min}\limits_{\pi} \leq 0 \Rightarrow \exist \pi, V^{\pi}_h \leq0$$意味着当前状态$s$一定存在一个策略可以实现硬安全

$$V^*_h (s)> 0 \Rightarrow \mathop{\min}\limits_{\pi} > 0 \Rightarrow \forall \pi, V^{\pi}_h > 0$$意味着当前状态$s$不存在任何策略可以实现硬安全

由于我们利用的是自车视角下的$V_h$,很多情况下无法获取其$V^{\pi}_h$,因为自车算法很多情况下不可知,且很多规则型的也无法训练,因此只能利用off-line学习的$V^*_h$来使用,甚至$V^*_h$更好,因为表示如果自车$V^*_h (s)\leq 0$但是还是撞了,说明是自车算法不是最优解导致的碰撞,最优解可以避免该碰撞.

**Definition of Advantage:**

$$Q^*_h(s,a):=\mathop{\min}\limits_{\pi}Q^{\pi}_h(s,a):=\mathop{\min}\limits_{\pi}\mathop{\max}\limits_{t \in \mathbb{N}}h(s_t), s_0=s,a_0=a,a_{t+1}\sim\pi(\cdot|s_{t+1})$$

$Q^*_h(s,a)$表示,当已经采取了动作$a$,之后一直采取最安全的策略,

$$A^*_h=Q^*_h-V^*_h$$,当其值>0时,代表当前采取的动作$a$提升了未来发生不可避免碰撞的概率,应该避免,所以要取负号

