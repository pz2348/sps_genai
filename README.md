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

## Assignment 4 · CIFAR-10 image generation

The Docker image includes the trained weights under `checkpoints/`. From the
project root, build and run the image.

```sh
docker build -t sps-genai:assignment4 .
docker run --rm -e OMP_NUM_THREADS=2 -p 127.0.0.1:8000:8000 sps-genai:assignment4
```

Both endpoints return an RGB PNG grid of generated 32 × 32 images. The
optional `seed` reproduces a request on the same runtime. The optional
`num_images` accepts 1 through 16 and defaults to 4.

```sh
curl --fail-with-body http://localhost:8000/generate-diffusion \
  -H 'Content-Type: application/json' \
  -d '{"num_images":4,"seed":42}' -o diffusion.png

curl --fail-with-body http://localhost:8000/generate-energy \
  -H 'Content-Type: application/json' \
  -d '{"num_images":4,"seed":42}' -o energy.png
```

To reproduce the packaged checkpoints, run one full epoch per model. The
training script downloads CIFAR-10 if needed and saves the weights under
`checkpoints/`. Increase `--epochs` for longer training.

```sh
uv sync --locked
uv run --no-sync python train_cifar_generators.py diffusion --epochs 1 --threads 3
uv run --no-sync python train_cifar_generators.py energy --epochs 1 --threads 3
```

The diffusion generator trains a timestep-conditioned U-Net to predict the
added Gaussian noise and draws samples using 100 reverse diffusion steps.
The energy generator uses a scalar CNN, a replay buffer for negative samples,
and 120 Langevin steps at inference. Sampling differentiates the scalar
energy with respect to the pixels. The outer contrastive training loss
updates the model parameters.
