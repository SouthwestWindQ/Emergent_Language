## Emergent Language (No-RL Branch)

本分支存储直接用分类任务建模游戏过程的训练与测试代码。具体逻辑可以见`gumbel_ae.py`。需要运行时，直接运行`python gumbel_ae.py`即可。

**模型修改**：所有model增加BatchNorm（事实证明可以显著提升效果）。outside agent的两个模型的一些细节略有调整。

**规则生成**：可以通过运行`generate_rules.py`生成。请注意：现成的checkpoints是用文件夹中`rule.json`的规则运行的。如你重新生成了一套规则，则需要重新训练。

**超参数**：程序中所使用的超参数都经过一定的调整。batch_size为16比为4好非常多，vocab_size为64比为27好非常多，lr为1e-4会好于其他学习率（lr为3e-4无法收敛，lr为5e-5可以收敛但效果不及1e-4，更低的学习率无法收敛）。

**测试**：可通过运行`python debug.py`测试现有的checkpoint。该checkpoint是20万个iteration中最好的：对所有的初始状态，正确率为25/27，结果是较为令人满意的。`debug.py`中的代码可以自行修改，如让它储存中间输出的symbol index等等。

**其它**：（1）simple文件夹中存储的是简化的游戏环境的测试结果，在这一简化环境中，inside agent通过symbol传递init state，outside agent直接通过这一symbol预测init state。simple文件夹中存储的是三个最好的checkpoint，它们对所有初始状态都达到了27/27的准确率。

（2）我们的环境看似简单，但对超参数的要求是比较高的。上面已经提到不同的lr与batch size会对结果有非常显著的影响，此外过度训练也会使表现不升反降（最好的checkpoint发生在20000-35000个iteration，再之后的iteration偶会出现表现结果较好的，但均不如最好的checkpoint）。

