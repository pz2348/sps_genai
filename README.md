# Module 9 · GPT-2 问答

在项目根目录运行以下命令。训练脚本从 Hugging Face 下载并缓存 Nectar 数据和 GPT-2
权重，打印微调前后的测试损失，并把模型和分词器保存到 `models/gpt2-nectar`。

```sh
uv sync --locked
uv run --no-sync python train_gpt2.py
```

默认使用 256 道单轮问题训练一轮，并留出 32 道问题测试。脚本用排名最高的回答作为
目标，只更新 GPT-2 最后两个 Transformer 层和最终的 LayerNorm。用
`uv run --no-sync python train_gpt2.py --help` 查看参数。如需另训一份检查点，
用 `--output models/gpt2-nectar-v2`，并相应修改下方的挂载来源。

启动 Docker Desktop 后，构建镜像并以只读方式挂载检查点。

```sh
docker build -t sps-genai:module9 .
docker run --rm -p 127.0.0.1:8000:8000 \
  --mount "type=bind,source=$PWD/models/gpt2-nectar,target=/app/models/gpt2-nectar,readonly" \
  sps-genai:module9
```

`start_word` 填完整问题，`length` 是最多生成的新 token 数。
接口在 `generated_text` 字段返回回答。

```sh
curl --fail-with-body http://localhost:8000/generate_with_llm \
  -H 'Content-Type: application/json' \
  -d '{"start_word":"Who wrote The Count of Monte Cristo?","length":40}'
```
