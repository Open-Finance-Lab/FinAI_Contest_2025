# Task 3 — DeFi Liquidity Provisioning

## Objective of the Task

Participants will develop intelligent agents for decentralized liquidity provisioning on Uniswap v3. The task simulates a Liquidity Provider (LP) who must decide, at each time step, whether to maintain or rebalance their liquidity position. This is framed as a reinforcement learning (RL) control problem, where the goal is to maximize trading fee revenue while minimizing impermanent loss (IL) and rebalancing (gas) costs.

The agents must learn to adjust price intervals dynamically in response to market changes, identifying optimal rebalancing strategies that outperform passive LP baselines. Participants are encouraged to focus on designing effective state representations, reward functions, and policies under uncertainty, using Deep RL or other sequential decision-making approaches.

This competition is based on the research paper:  
*"Improving DeFi Accessibility through Efficient Liquidity Provisioning with Deep Reinforcement Learning"* ([arXiv:2501.07508](https://arxiv.org/abs/2501.07508)).  
An accompanying codebase implementing the baseline strategy and environment is available here:  
https://github.com/Alessiobrini/deeprl-liquidity-provision-uniswapv3

Participants may use this implementation as a starting point for their solutions.

---

## Datasets to Be Provided

### Training Dataset

Hourly data for the WETH/USDC Uniswap v3 pool (0.05% fee tier), covering the period from **May 5, 2021 at 1:00 am to January 29, 2024 at 7:00 pm**. The training dataset ends on January 29, 2024 at 7:00 pm. Participants may use their own train/validation split strategies, but must avoid relying on data beyond this timestamp. The dataset is sourced from the Uniswap Ethereum subgraph and resampled into an evenly spaced hourly series.

Participants will be provided with the market price time series and a starter reinforcement learning environment (`Uniswapv3Env`) that constructs the state vector automatically at each time step. The environment internally computes the following state variables:

- **Spot price**
- **Tick index**  
  (Mapped from price using log spacing: *i = floor[ log(pₜ) / log(1.0001) ]*)
- **Liquidity interval width**  
  (Selected in the previous step by the agent)
- **Liquidity level**  
  (Capital deployed, computed from the AMM formula given the selected price range)
- **Exponentially weighted volatility**  
  (Estimated using α = 0.05)
- **Moving averages**:
  - 24-period MA (short-term trend)
  - 168-period MA (long-term trend)
- **Bollinger Bands**  
  (Upper, middle, and lower bands using T3 smoothing)
- **Technical indicators computed with TA-Lib**:
  - ADXR (Average Directional Movement Index Rating)
  - BOP (Balance of Power)
  - DX (Directional Movement Index)

### Testing Dataset (Held-Out)

While participants may validate using any strategy of their choice, final evaluation will be conducted on a private test window:
January 29, 2024 at 8:00 pm to April 29, 2024 at 7:00 pm, which is not included in the released dataset.
Participants should take care not to introduce lookahead bias when preparing features or targets, and must ensure that only past and current information is used at each time step during training and evaluation.

### Environment Code

A simplified RL environment will be provided based on the codebase at  
https://github.com/Alessiobrini/deeprl-liquidity-provision-uniswapv3,  
using the Gymnasium interface and Stable Baselines3 PPO implementation.

---

## What Participants Should Submit

- Complete code for training and inference (scripts, classes, and configurations)
- Trained model weights
- An evaluation script (e.g., `evaluate_agent.py`) that runs inference on the provided test environment on the test dataset (**January 29, 2024 at 8:00 pm to April 29, 2024 at 7:00 pm**)
- A short README file with setup and usage instructions
- Optionally: Docker container with all dependencies for reproducibility

---

## Evaluation Metrics and Methodology

For leaderboard scoring, submissions will be evaluated on a **fixed held-out testing period** as described above.

The evaluation will focus on the **cumulative performance** of the agent over this entire test window.

The main metric is the **cumulative reward**, computed at each time step as:

> **Reward = Trading Fees − LVR Penalty − Gas Cost**

Where:
- **Trading Fees** are collected when swaps occur within the active price range;
- **LVR Penalty** (Loss-Versus-Rebalancing) represents the opportunity cost of capital locked in the AMM compared to alternative uses;
- **Gas Cost** is incurred when rebalancing the liquidity position.

The agent's performance will be compared against a **passive LP baseline**, which rebalances its price range at fixed 500-hour intervals without adapting to market dynamics.

**Only the final cumulative reward over the evaluation window will be used to rank submissions.**

---

## Additional Requirements or Assumptions

### Action Space

Agents must operate in a **discrete action space**, choosing from predefined tick-width intervals (e.g., {0, 20, 50}). While the action space must remain discrete, **participants are free to customize** the number and values of tick-width options provided to the agent if they believe this improves performance.

### Environment and Reward Function

The environment implementation provided in the starter kit **must remain unchanged**. In particular:

- The **reward function must not be modified**. It is already implemented in the environment and reflects the net value of an LP position after fees, losses, and gas costs.
- The **gas fee is fixed at $5 per rebalancing** and must remain unchanged.
- The initial amount of token X (base asset) is fixed at 2 units at the beginning of each episode and must not be altered to a different amount.

### Feature Engineering and External Data

Participants are encouraged to extend the **state (feature) space**, including integrating external data such as mempool snapshots or centralized exchange prices. However, they must:

- Preserve the **existing state variables from the baseline as a subset** of their feature space.
- Ensure **reproducibility** and clearly document any external data used.

### No Lookahead Bias

Agents may only use information available at the current or past timesteps.  
**Future information from the test set is strictly prohibited.**


## Repository Structure and Usage

The starter kit is organized into the following components:

```
.
├── config/                    # YAML files to control hyperparameters and environment settings
├── custom_env_folder/        # Contains the UniswapV3 RL environment (compatible with Gymnasium)
├── data_price_uni_h_time.csv # Hourly WETH/USDC Uniswap v3 data (used by both training and evaluation scripts)
├── requirements.txt          # List of required Python packages
├── uniswap_test.py           # Main training script using PPO and the custom environment
├── uniswap_test_bm.ipynb     # Notebook running the passive LP baseline for comparison
```

- **Training a Custom Agent**: Use `uniswap_test.py` to train agents in the provided environment. This script leverages the custom Gym-compatible environment found in `custom_env_folder/`.

- **Configuration Management**: The `config/` folder contains YAML files to specify:
  - PPO hyperparameters (e.g., learning rate, discount factor)
  - Design choices (e.g., action space, environment parameters)

- **Baseline Comparison**: The notebook `uniswap_test_bm.ipynb` runs a passive liquidity provisioning strategy that rebalances at fixed time intervals. It serves as the benchmark to beat.

- **Dataset**: The file `data_price_uni_h_time.csv` contains the hourly price series for the ETH/USDC pool on Uniswap v3. This is the only market data used in the baseline.

- **Dependencies**: You can install the required packages using:
  ```bash
  pip install -r requirements.txt
  ```

