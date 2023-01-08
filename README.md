# tinkoff_dl_antiplagiat</br>
Утилита для проверки текстов программ с расширением .py на плагиат</br>
### Структура проекта</br>
tinkoff_dl_antiplagiat</br>
|—data</br>
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|—files</br>
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|—file1.py</br>
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|—plagiat1</br>
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|—file1.py</br>
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|—plagiat2</br>
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|—file1.py</br>
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|—input.txt</br>
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|—score.txt</br>
|—models</br>
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|—model1.pkl</br>
|—src</br>
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|—train.py</br>
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|—compare.py</br>
### Запуск</br>
Установка зависимостей</br>
``` pip install -r requirements.txt```</br>
Для обучения необходимо иметь cudatoolkit для pytorch</br>
Обучение модели (cuda)</br>
```python train.py ../data/files ../data/plagiat1 ...data/plagiat2 --model ../models/model.pt```</br>
Обучает модель на файлах из папки files и папок plagiat1, plagiat2 и сохраняет ее в папку models</br>
Проверка текста на плагиат</br>
```python compare.py ../data/input.txt ../models/model.pt```</br>
Загружает модель из папки models и проверяет пары текстов из файла input.txt на плагиат</br>
Формат файла input.txt</br>
```file1.py file2.py```</br>
```file1.py file3.py```</br>
```file2.py file3.py```</br>
Результат в файле score.txt</br>
```0.5```</br>
```0.7```</br>
```0.9```</br>
0 Соответствует 0% плагиата, 1 - 100% плагиата</br>
### Описание модели</br>
В Dataset подаются директории, содержащие файлы для обучения.</br>
После определения файлов, доступных для обучения, они преобразуются в обучающие параметры.</br>
#### Параметры:</br>
* Отношение длины текстов</br>
* Среднее расстояние Левенштейна между ближайшими по расстоянию импортами</br>
* Среднее расстояние Левенштейна между ближайшими по расстоянию именами классов</br>
* Среднее расстояние Левенштейна между ближайшими по расстоянию именами функций</br>

Параметры подаются в нейронную сеть, состоящую из 3 линейных слоев. В качестве функции активации используется ReLU, т.к. отношение длин и расстояния неотрицательные</br>
Конечная вероятность определяется сигмоидной функцией. В качестве функции потерь используется бинарная кросс-энтропия</br>
Размер батча определяется из необходимости иметь как минимум 1 негативный пример (плагиат) на батч. При 306 файлах для обучения есть ~47000 пар без плагиата. При 612 парах с плагиатом хотя размер батча должен быть больше 80</br>
