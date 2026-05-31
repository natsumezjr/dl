# ConvOnlyCaptcha 可视化输出说明

## 准确率

- 验证集 loss：`3.049005`
- 字符级准确率 char_accuracy：`29.6700%`
- 整体验证码准确率 captcha_accuracy：`0.5800%`

## 文件说明

- `00_sample_image.png`
- `01_learned_conv_kernels.png`
- `02_activation_conv_raw.png`
- `03_activation_signed_positive_negative.png`
- `04_logits_heatmap.png`
- `05_learned_kernel_0_scanning.gif`

## 注意

- 这个模型没有 ReLU，所以 `02_activation_conv_raw.png` 展示的是卷积层原始输出。
- `03_activation_signed_positive_negative.png` 用冷暖色显示正负响应。
- 如果准确率很低，这是预期现象之一：只有卷积 + 全连接整体表达能力有限，而且没有非线性激活、池化和归一化。