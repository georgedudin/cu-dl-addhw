# Report

## Track

Выбранный трек:

```text
B (Small GPU, Mac M4 Pro 24GB unified, MPS)
```

## Что реализовано

- [x] dataset.py
- [x] processor.py
- [x] model.py
- [x] train.py
- [x] benchmark.py

## Конфигурация

```text
config path: configs/track_b_small_gpu_medium.yaml
seed: 42
device: mps
dtype: float32
max_steps: 500
batch size: 4
lr: 5e-5
vision: google/vit-base-patch16-224 (frozen)
llm: Qwen/Qwen2-0.5B-Instruct (frozen)
adapter: Linear(768, 896) + GELU + Linear(896, 896), avg pool 197 -> 49
num_image_tokens: 49
max_length: 384
train: assets/math_vqa_medium/manifest.jsonl, split=train, 180 примеров
```

## Результаты

```text
public tests: 14 passed
train loss: 6.4 на старте, к концу ~0.5, NaN не было
benchmark (dev, 40 примеров):
  overall:                 0.20
  subject/algebra:         0.40
  subject/coordinate_geom: 0.00
  subject/geometry:        0.00
  subject/linear_algebra:  0.00
  subject/plots:           0.60
```

## Использованные ресурсы

```text
Apple M4 Pro, MPS
память: ~5-7 GB unified
обучение: ~4 минуты на 500 шагов (~2 it/s)
benchmark: ~8 сек на 40 примеров
```

## Анализ ошибок

8 правильных из 40. Три ошибки:

1. medium_dev_0003 (geometry, gold=B): "Прямоугольник 5x2, площадь?". Модель вывела "\nassistant: 10". Само число верное (5*2=10, вариант B), но не буквой, и парсер ничего не достал.

2. medium_dev_0000 (algebra, gold=D): "y=1x+3, y при x=2?". Вывод "\ned:" - просто шум, продолжение prompt-а. По plots модель работает в 3 раза лучше, видимо адаптер ловит сигнал с графиков, но не с формул.

3. medium_dev_0004 (geometry, gold=A): "Треугольник с катетами 9 и 12, гипотенуза?". Тоже "\ned:". На geometry стабильно 0% - адаптер не вытягивает из ViT нужные геометрические признаки.

## Комментарии

Больше всего возни было с MPS. Сначала на adapter.pre_norm.weight градиент уходил в NaN на первом же шаге, причём остальные параметры были нормальные. Выкинул LayerNorm из адаптера, оставил два линейных слоя - стало стабильно.

Потом adaptive_avg_pool1d на MPS не работает когда 197 не делится на 49, тоже падало. Заменил на avg_pool1d с kernel=5 stride=4 - даёт ровно 49 групп.

И ещё HF generate с inputs_embeds вёл себя странно - постоянно выдавал токен 0 ("!") и сразу EOS. Forward при этом работал нормально, в top-1 на позиции ответа была правильная буква. Дописал свой greedy цикл с kv-cache, после этого accuracy уехала с 0% до 20%.

float16 на MPS не использовал, на Qwen2 даёт NaN в RMSNorm. Float32 хватает с запасом.

Что бы сделал по-другому:
- больше данных, 180 примеров на 5 тем мало
- в prompt явно требовать однобуквенный ответ, модель иногда уходит в свободный формат
- попробовать LoRA на attention LLM, чисто адаптер слабоват

## Критерии оценивания

См. файл [`GRADING.md`](GRADING.md).
