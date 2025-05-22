# NN from Scratch

Ручная реализация базовых компонентов нейросети с использованием numpy и scipy (без torch),
и дальнейшее сравнение с `torch.nn`.  
Задание взято из курса Practical DL (ШАД).

## Цель:

- Глубже понять, как устроены слои на уровне numpy
- Реализовать forward- и backward-проходы вручную

---

## Реализованные модули

### Слои
- `Linear`
- `Conv2D` 
- `MaxPool2D` 
- `BatchNorm` 
- `Dropout` 


- `Flatten`, `Sequential` — инструменты для сборки модели

### Активации
- `ReLU`, `LeakyReLU`
- `ELU`, `SoftPlus`
- `Softmax`, `LogSoftmax`

### Функции потерь
- `MSECriterion`
- `ClassNLLCriterionUnstable` (нестабильная)
- `ClassNLLCriterion` (численно стабильная)

### Оптимизаторы
- `SGD` с momentum
- `Adam` с bias correction

---

## Тесты

Модули протестированы и сравниваются с PyTorch аналогами (`torch.nn` и `torch.nn.functional`).