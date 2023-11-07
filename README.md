# RESA: Reasonable Evaluation via Safety Assurance

# Content

* [Train Agent](#Train-Agent)
* [Train Scenario](#Train-Scenario)
* [Train Safety network](#Train-Safety-network)
* [Eval](#Eval)
* [Visualization](#Visualization)

## Train Agent

### Usage

``````bash
# Launch CARLA
./CarlaUE4.sh -prefernvidia -windowed -carla-port=2000

# Launch in another terminal
python scripts/run.py --agent_cfg behavior.yaml --scenario_cfg standard_eval.yaml --mode train_agent
``````

### Policy

* **behavior:** Carla default agent (no state)

* **expert:** Carla leaderboard default rule-based agent (ego state)
* **plant:**  transformer based planning agent, output ego's future waypoints (ego state)
* **sac:** RL-based agent (from Safebench, default 4dim simple state)

### state/observation

For learnable agent (SAC PPO......)

* **Safebench default state for learning**

  1. **lateral dis** from the target point

  2. **-delta yaw** from the target point

  3. **ego speed**

  4.  **front got vehicle or not**  

     shape: 4

* **PlanT style encoded state**

  1. cls token after transformer block

     shape: [1, 1, 512]

For Carla Leaderboard agent (Expert or PlanT)

* **Ego state for PlanT and Expert**:
  1. ego position (x, y)
  2. ego speed (m/s)
  3. ego yaw (radius)

## Train Scenario

### Usage

```bash
# Launch CARLA
./CarlaUE4.sh -prefernvidia -windowed -carla-port=2000

# Launch in another terminal
python scripts/run.py --agent_cfg behavior.yaml --scenario_cfg standard_eval.yaml --mode train_scenario
```

### Policy

1. **sac**
2. **standard (autopilot)**
3. **ppo**
4. **td3**

### optional

* Select the controlled bv method

  1. **attention-based** (default)
  2. **rule-based**

```bash
python scripts/run.py --agent_cfg behavior.yaml --scenario_cfg standard_eval.yaml --mode train_scenario  --cbv_selection 'attention-based'  # different method of selecting controlled bv
```

* Input state

  **Actor info** (ego and surrounding 3 bv's 9-dim state)

* Output action

  2-dim continues action

  acc ~ [-3.0, 3.0]

  steer ~ [-0.3, 0.3]

## Train Safety network

### Usage

```bash
# Launch CARLA
./CarlaUE4.sh -prefernvidia -windowed -carla-port=2000

# Launch in another terminal
python scripts/run.py --agent_cfg behavior.yaml --scenario_cfg standard_eval.yaml --safety_network_cfg HJR.yaml --mode train_safety_network
```

### Policy

* HJ-Reachability

### Input state

[yaml file](safebench/safety_network/config/HJR.yaml)  ```obs_type: "plant" (default)```

* **PlanT:** (encoded state) default 

* **Actor info:** (ego and surrounding 3 bv's 9-dim state) *no route map information*

## Eval

### Usage

```bash
# Launch CARLA
./CarlaUE4.sh -prefernvidia -windowed -carla-port=2000

# Launch in another terminal
python scripts/run.py --agent_cfg behavior.yaml --scenario_cfg standard_eval.yaml --safety_network_cfg HJR.yaml --mode eval
python scripts/run.py --agent_cfg behavior.yaml --scenario_cfg standard_eval.yaml --safety_network_cfg HJR.yaml --mode eval --safety_eval  # use the trained safety network to help evaluation
```

## Visualization

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
* out of road length
* distance to route
* incomplete route
* running time
* final score
