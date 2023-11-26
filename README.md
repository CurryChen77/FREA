# RESA: Reasonable Evaluation via Safety Assurance

# Content

* [Eval](#Eval)

* [Train Agent](#Train-Agent)
* [Train Scenario](#Train-Scenario)
* [Train Safety network](#Train-Safety-network)
* [Visualization](#Visualization)

## Eval

### Desktop Users

```bash
# Launch CARLA
./CarlaUE4.sh -prefernvidia -windowed -carla-port=2000

# Launch in another terminal
python scripts/run.py --agent_cfg behavior.yaml --scenario_cfg standard_eval.yaml --safety_network_cfg HJR.yaml --mode eval --render
python scripts/run.py --agent_cfg behavior.yaml --scenario_cfg standard_eval.yaml --safety_network_cfg HJR.yaml --mode eval --safety_eval --render # use the trained safety network to help evaluation
```

### Remote Users

#### Remote training

```bash
# Launch CARLA with headless mode
./CarlaUE4.sh -prefernvidia -RenderOffScreen -carla-port=2000

# Another terminal no display pygame
SDL_VIDEODRIVER="dummy" python scripts/run.py --agent_cfg plant.yaml --scenario_cfg sac.yaml --mode train_scenario

# Showing the memory usage
SDL_VIDEODRIVER="dummy" mprof run --python scripts/run.py --agent_cfg plant.yaml --scenario_cfg sac.yaml --mode train_scenario
```

#### Visualize pygame window

```bash
# Launch CARLA with headless mode
./CarlaUE4.sh -prefernvidia -RenderOffScreen -carla-port=2000

# Another terminal, Launch on the virtual display
DISPLAY=:10 python scripts/run.py --agent_cfg behavior.yaml --scenario_cfg standard_eval.yaml --mode eval --render
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
python scripts/run.py --agent_cfg behavior.yaml --scenario_cfg standard_train.yaml --mode train_agent
``````


### Policy

* **behavior:** Carla default agent (no state)

* **expert:** Carla leader board default rule-based agent (ego state)
* **plant:**  transformer based planning agent, output ego's future waypoints (ego state)
* **sac:** RL-based agent (from Safebench, default 4 dim simple state)

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

For Carla Leader-board agent (Expert or PlanT)

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
python scripts/run.py --agent_cfg behavior.yaml --scenario_cfg sac.yaml --mode train_scenario
```

### Policy

1. **sac**
2. **standard (autopilot)**
3. **ppo**
4. **td3**

### optional

* Select the CBV method **based on the CBV candidate**

  1. **attention-based** (default, attention weight)
  2. **rule-based** (nearest)

```bash
python scripts/run.py --agent_cfg behavior.yaml --scenario_cfg standard_eval.yaml --mode train_scenario  --cbv_selection 'attention-based'  # different method of selecting controlled bv
```

* Input state

  **Actor info**

  |  vehicle   |  x   |  y   | yaw  | cos (yaw) | sin (yaw) |  Vx  |  Vy  | Acc.x | Acc.y |
  | :--------: | :--: | :--: | :--: | :-------: | :-------: | :--: | :--: | :---: | ----- |
  | CBV state  |      |      |      |           |           |      |      |       |       |
  | ego state  |      |      |      |           |           |      |      |       |       |
  | BV_1 state |      |      |      |           |           |      |      |       |       |
  | BV_2 state |      |      |      |           |           |      |      |       |       |
  | BV_3 state |      |      |      |           |           |      |      |       |       |

  **Tips: if no BV then the corresponding state will set to 0**

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
python scripts/run.py --agent_cfg behavior.yaml --scenario_cfg standard_train.yaml --safety_network_cfg HJR.yaml --mode train_safety_network
```

### Policy

* HJ-Reachability

### Input state

[yaml file](safebench/safety_network/config/HJR.yaml)  ```obs_type: "ego_info" (default)```

* **PlanT:** (encoded state) 

* **Ego info:** (ego and surrounding 4 meaningful BVs' 9-dim state) *no route map information* (default)

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

## Tricks

### Traffic light

* Ego vehicle's front traffic

  When the ego vehicle just run out of the junction, find the new closest traffic light, and set the corresponding light state to Green (the first traffic light has already set to Green at initialization)

* CBV front traffic

  When the ego is in the junction, find whether the selected CBV in under the control of Red traffic light, if so, set the affecting traffic light state to Green

### CBV selection candidate

* Except the ego vehicle
* Except the BVs which are in the opposite lane side, but on the same road
* Except the BVs behind the ego vehicle and the abs yaw angle difference > 45 degree
