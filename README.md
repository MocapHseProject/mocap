### Подготовка к запуску
 https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html -- для запуска проекта требуется наличие установленной версии CUDA не ниже 10й версии и операционная система Windows (не ниже 10й версии)
Версия Java не ниже 9й
### Описание проекта
Глобально наш проект делится на несколько частей:
1) Получение видеопотока + инициализация модели
2) Его обработка (фильтрация каждого кадра видеопотока для выявления положений маркеров)
3) Передача полученный позиций суставов модели (в текстурных координатах)
4) Расчет позиции суставов в 3-х мерном пространстве
5) Работа рендера
  5.1) Анимация модели
  5.2) Её отрисовка
Для реализации нашей идеи нам необходима библиотека для работы с 3х мерной графикой: мы используем OpenGL + glsl тк данная библиотека кроссплатформенная и не слишком навороченная (тк нам нужны только базовые ресурсы рендера), CUDA для обработки видеопотока на GPU: на видеокарте тк кадры обрабатываем попиксельно (альтернатива compute шейдера – не хотим), gRPC чтобы связать CUDA и OpenGL (тк CUDA на C++ а рендер на JAVA) + нам нужен формат 3D моделей, которые мы хотим анимировать: COLLADA, тк анимации можно писать только под модели со скелетом