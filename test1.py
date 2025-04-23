import torch

# 超参数设置
CLIP_PARAM = 0.2

# 手动设定新旧对数概率
new_log_probs = torch.tensor([-0.3, -0.1, -0.3, -1])
old_log_probs = torch.tensor([-2, -2, -2, -0.1])

# 手动设定优势函数值，前两个为正，后两个为负
batch_advantages = torch.tensor([1, 1, -1, -1])

# 计算新对数概率与旧对数概率的比值的指数
ratio = torch.exp(new_log_probs - old_log_probs)

# 计算第一个替代损失
surr1 = ratio * batch_advantages

# 对比例进行裁剪，并计算第二个替代损失
clipped_ratio = torch.clamp(ratio, 1 - CLIP_PARAM, 1 + CLIP_PARAM)
surr2 = clipped_ratio * batch_advantages

# 取两个替代损失的最小值的负均值作为策略损失
policy_loss = -torch.min(surr1, surr2).mean()

print("原始 ratio:", ratio)
print("裁剪后的 ratio:", clipped_ratio)
print("第一个替代损失 surr1:", surr1)
print("第二个替代损失 surr2:", surr2)
print("选择的替代损失:", torch.min(surr1, surr2))
print("策略损失:", policy_loss.item())