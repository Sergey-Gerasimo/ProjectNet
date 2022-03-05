# О чем проект?

Данный проект - первая ступень в разработке алгоритма восстановления цвета в изображениях. Данный проект подразумевает анализ существующих моделий сети и, конечно, разработка программног кода, реализирующих данные модели (открытый код для моделей сложно найти да и не зачем)

# Зачем? 

Те модели, которые были рассмотеры, не в состоянии восстанавливать изображения с разным разрешением(возможно обучить модель под определенное разрешение, при условии удовлетворяющей ошибки).

# Реализация 

В моем проекте есть 2 основные версии для кадой сети. Первая модель полностью совметима только с TensorFlow2, а вторая условно совметсим и с Tensorflow2 и с Tensorflow1. Исключением состовляет генеративно состязательная модель. Для этой модели мне не удалось реализовать работающий алгоритм обучения.

### **TensorfFlow2 совметсимая модель**

Все классы находятся в модуле colorize.
В данном модуле реализованы классы:
1. FusinoLayer(inputs:list) -- слой конкатенирования тензоров
2. Folder(path:str, lenght:int) -- итерируемый класс выборки(по умалчанию длина равна количеству файлов в папке)
3. AutoEncoder(encoder, decoder, classifier=None) -- Модель сети автоэнкодера. Параметр classifier -- сеть классификатора(в моем примере используется VGG19) и является необязательным.
4. saved_network -- сохраняемая модель сети
5. GAN -- генеративно состязательная модеь сети 
6. VGG19 -- модель сети VGG19
7. функция write -- функция вывода информации 
8. функция перевода формата LAB в RGB


### **Tenosflow1/2 совметимая модель**
Все классы нходятся в модуле models. 
В данном модуле реализованя следующие классы: 
1. Block -- слой блок сети(например блок автоэнкодера)
2. FusinoLayer(inputs:list) -- слой конкатенирования тензоров
3. VGG19 -- сеть vgg-19, представленная в виде слоя сети 
4. VAE -- сеть вариационного автоэнкодера(аналогичный класс сети AutoEncoder)
5. Folder -- тот же самый класс, что и Folder в Tensorflow2 совместимой модели 
6. функция write -- функция вывода информации 
7. функция перевода формата LAB в RGB 

# Струтура проекта 

В проекте следующие папки:

1. model0 -- в этой папке веса первой модели
2. model1 -- в этой папке веса вторй модели 
3. TF2 -- в этой папке код TensorfFlow2 совместимых моделей 
4. TF1 -- в этой папке код TensorfFlow1/2 совместимых моделей

Также в проекте есть следующие файлы: 

1. tmp1.txt -- история обучения первой сети 
2. tmp2.txt -- история обучения второй сети 
3. tranzit.py -- Python скрипт, который преобразует данные из архитектуры выборки, подобной ImageNet в необходимую структуру и сжимает изображения до разрешения 256x256

# предполажения по архитектуре алгоритма 

Все рассмотренные модели обладают существенным неостатком -- они рассчитаны под конктретное разрешение изображения. Неменине выжным недостатком является то, что такие модели плохо работают с изображениями с большим количеством различных объектов. 

Последнюю проблему можно решить с помощью сегментации изображения на различные объекты. То есть, вместо сети классификатора, мы строим модель, которая может выделять объект и расспазновть его (такие сети уже существуют). Каждый объект мы вырезаем и сохраняем как отдельное изображение и каждое из таких изображений подем на вход сети автоэнкодера, предваритльно сохранив класс объекта. При таком алгоритме сложность вычислений напрямую зависти от количества объектов на изображении.

Первую проблему решить существенно сложнее. Есть всего две догадки как можно такую проблему решить: 

1. сжимать изображение, а ответ сети увеличивать с помощью другой сети 
2. выделение сегмента изображение и представление его в качестве входных данных (подобно сверточным соям)

На данный момент я придерживюсь второг варианта ввиду его простоты
