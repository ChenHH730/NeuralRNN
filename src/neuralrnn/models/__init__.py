"""模型库（model zoo）。

为保持可选重依赖的懒加载，本包不在导入时加载各模型模块；请：
  - 跨家族统一构造：用 neuralrnn.AutoModel / AutoConfig（推荐）；
  - 或显式导入具体家族：from neuralrnn.models.ctrnn import CTRNNModel, CTRNNConfig。

已内置（参考实现）：ctrnn（范式A）、plrnn（范式B）、latent_circuit（范式A）、lowrank（范式A）。
待移植：lfads（见 docs/PORTING_GUIDE.md）。
"""
