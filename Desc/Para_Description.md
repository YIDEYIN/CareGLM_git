# 训练

## 命令

```bash
python src/finetune.py
```

## 参数

### class utils.config.ModelArguments [\<source\>](https://github.com/WangRongsheng/MedQA-ChatGLM/blob/main/MedQA-ChatGLM/utils/config.py#L27)

- **model_name_or_path** (str, *optional*): 预训练模型的路径或 [huggingface.co/models](https://huggingface.co/models) 的项目标识符。缺省值：`CHATGLM_REPO_NAME`
- **config_name** (str, *optional*): 预训练配置文件名称或路径，不指定则与 model_name 相同。缺省值：`None`
- **tokenizer_name** (str, *optional*): 预训练分词器名称或路径，不指定则与 model_name 相同。缺省值：`None`
- **cache_dir** (str, *optional*): 保存从 [huggingface.co](https://huggingface.co) 下载内容的文件夹路径。缺省值：`None`
- **use_fast_tokenizer** (bool, *optional*): 是否使用快速分词器。缺省值：`True`
- **model_revision** (str, *optional*): 将要使用的预训练模型版本。缺省值：`CHATGLM_LASTEST_HASH`
- **use_auth_token** (str, *optional*): 是否使用根据 `huggingface-cli login` 获取的认证密钥。缺省值：`False`
- **quantization_bit** (int, *optional*): 模型量化等级。缺省值：`None`
- **checkpoint_dir** (str, *optional*): 存放模型断点和配置文件的文件夹路径。缺省值：`None`
- **reward_model** (str, *optional*): 存放奖励模型断点的文件夹路径。缺省值：`None`

### class utils.config.DataTrainingArguments [\<source\>](https://github.com/WangRongsheng/MedQA-ChatGLM/blob/main/MedQA-ChatGLM/utils/config.py#L78)

- **dataset** (str, *optional*): 将要使用的数据集名称，使用英文逗号来分割多个数据集。缺省值：`alpaca_zh`
- **dataset_dir** (str, *optional*): 存放数据集文件的文件夹路径。缺省值：`data`
- **split** (str, *optional*): 在训练和评估时使用的数据集分支。缺省值：`train`
- **overwrite_cache** (bool, *optional*): 是否覆盖数据集缓存。缺省值：`False`
- **preprocessing_num_workers** (int, *optional*): 数据预处理时使用的进程数。缺省值：`None`
- **max_source_length** (int, *optional*): 分词后输入序列的最大长度。缺省值：`512`
- **max_target_length** (int, *optional*): 分词后输出序列的最大长度。缺省值：`512`
- **max_samples** (int, *optional*): 每个数据集保留的样本数，默认保留全部样本。缺省值：`None`
- **num_beams** (int, *optional*): 评估时使用的 beam 数，该参数将会用于 `model.generate`。缺省值：`None`
- **ignore_pad_token_for_loss** (bool, *optional*): 在计算损失时是否忽略填充值。缺省值：`True`
- **source_prefix** (str, *optional*): 在训练和评估时向每个输入序列添加的前缀。缺省值：`None`

### class utils.config.FinetuningArguments [\<source\>](https://github.com/WangRongsheng/MedQA-ChatGLM/blob/main/MedQA-ChatGLM/utils/config.py#L161)

- **finetuning_type** (str, *optional*): 训练时使用的微调方法。缺省值：`lora`
- **num_layer_trainable** (int, *optional*): Freeze 微调中可训练的层数。缺省值：`3`
- **name_module_trainable** (str, *optional*): Freeze 微调中可训练的模块类型。缺省值：`mlp`
- **pre_seq_len** (int, *optional*): P-tuning v2 微调中的前缀序列长度。缺省值：`16`
- **prefix_projection** (bool, *optional*): P-tuning v2 微调中是否添加前缀映射层。缺省值：`False`
- **lora_rank** (int, *optional*): LoRA 微调中的秩大小。缺省值：`8`
- **lora_alpha** (float, *optional*): LoRA 微调中的缩放系数。缺省值：`32.0`
- **lora_dropout** (float, *optional*): LoRA 微调中的 Dropout 系数。缺省值：`0.1`
- **lora_target** (str, *optional*): 将要应用 LoRA 层的模块名称，使用英文逗号来分割多个模块。缺省值：`query_key_value`
- **resume_lora_training** (bool, *optional*): 若是，则使用上次的 LoRA 权重继续训练；若否，则合并之前的 LoRA 权重并创建新的 LoRA 权重。缺省值：`True`
- **plot_loss** (bool, *optional*): 微调后是否绘制损失函数曲线。缺省值：`False`

### class utils.common.Seq2SeqTrainingArguments [\<source\>](https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/training_args_seq2seq.py#L30)

这里仅列出部分关键参数，详细内容请查阅 [HuggingFace Docs](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments)。

- **output_dir** (str): 输出模型权重和日志的文件夹路径。
- **overwrite_output_dir** (bool, *optional*): 是否覆盖输出文件夹。缺省值：`False`
- **do_train** (bool, *optional*): 是否执行训练。缺省值：`False`
- **do_eval** (bool, *optional*): 是否执行评估。缺省值：`False`
- **do_predict** (bool, *optional*)：是否执行预测。缺省值：`False`
- **per_device_train_batch_size** (int, *optional*): 用于训练的批处理大小。缺省值：`8`
- **per_device_eval_batch_size** (int, *optional*): 用于评估或预测的批处理大小。缺省值：`8`
- **gradient_accumulation_steps** (int, *optional*): 梯度累加次数。缺省值：`1`
- **learning_rate** (float, *optional*): [AdamW](https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/optimizer_schedules#transformers.AdamW) 优化器的初始学习率。缺省值：`5e-5`
- **weight_decay** (float, *optional*): [AdamW](https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/optimizer_schedules#transformers.AdamW) 优化器除偏置和归一化层权重以外使用的权重衰减系数。缺省值：`0.0`
- **max_grad_norm** (float, *optional*): 梯度裁剪中允许的最大梯度范数。缺省值：`1.0`
- **num_train_epochs** (float, *optional*): 训练轮数（若非整数，则最后一轮只训练部分数据）。缺省值：`3.0`
- **logging_steps** (int, *optional*): 日志输出间隔。缺省值：`500`
- **save_steps** (int, *optional*): 断点保存间隔。缺省值：`500`
- **no_cuda** (bool, *optional*): 是否关闭 CUDA。缺省值：`False`
- **fp16** (bool, *optional*): 是否使用 fp16 半精度（混合精度）训练。缺省值：`False`
- **predict_with_generate** (bool, *optional*): 是否生成序列用于计算 ROUGE 或 BLEU 分数。缺省值：`False`

# 推理

## 命令

```bash
python src/infer.py
```

## 参数

- **checkpoint_dir** (str, *optional*): 存放模型断点和配置文件的文件夹路径。缺省值：`None`
