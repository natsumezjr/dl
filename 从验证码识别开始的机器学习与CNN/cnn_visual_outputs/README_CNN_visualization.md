# CNN 可视化输出说明

样本图片：`D:\大学文件夹\这学期的课题\机器智能与信息对抗\实验\3-2 CNN验证码识别\captcha_cnn_project\data\train\0164_010538.png`  
是否为 fallback 生成：`False`

## 文件说明

- `01_fixed_kernels_response.png`：固定卷积核示例。展示横线、竖线、边缘等简单卷积核扫过验证码后产生的响应图。
- `02_convolution_scanning.gif`：动态卷积过程。红框表示 3x3 kernel 当前覆盖的局部窗口，右侧逐步填充响应图。
- `03_pooling_effect.png`：MaxPool 的直观效果。它保留局部窗口内最强响应，同时降低空间分辨率。
- `04_model_input_and_logits.png`：如果成功加载训练模型，则展示输入图和最终 logits，logits 形状是 `[4, 36]`。
- `05_activation_*.png`：训练好的 CaptchaCNN 中间层通道激活图。
- `06_layer_mean_response_progression.png`：从浅层到深层的平均响应图，观察空间尺寸逐步下降、通道语义逐步增强。
- `cnn_visualize_layers.py`：可重复运行的完整脚本。

## 直觉

固定卷积核说明：卷积不是玄学，它就是“局部窗口”和“模板”的逐元素乘积求和。  
训练好的 CNN 只是把这些模板从手写固定值，变成从数据中自动学习出来的参数。
