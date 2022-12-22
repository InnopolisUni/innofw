Для использования Фреймворка для поддержки принятия решений в функционировании системы безопасности предприятия необходимо выполнить следующие шаги:
1.	Склонировать фреймворк;

    ```git clone https://github.com/InnopolisUni/innofw.git```

2.	Установить пакеты poetry;

    ```poetry install```

3.	Для инициализации модели предобученными весами указать путь к весам в параметре ckpt_path эксперимента, либо в shell/batch скрипта;
4.	Путь к наборам данных указан в параметре source конфигурационных файлов datasets (набор данных загрузится автоматически при запуске скрипта);
5.	Запустить алгоритмы посредством shell/batch скриптов.

Команды с использованием shell скриптов приведенные ниже должны быть использованы на ОС Линукс
Команды с использованием batch скриптов приведенные ниже должны быть использованы на ОС Windows 

Список алгоритмов:
1)	интеллектуальный анализ и мониторинг потоков данных, система сетевого видеонаблюдения для обеспечения видеоаналитики и работы с потоками данных (ONVIF)
    
    Пример использования (sh/bat скрипты):
    - linux:
        - ```sh camera_info.sh```
        - ```sh stream.sh```
        - ```sh mover_pan_left.sh```
        - ```sh mover_pan_right.sh```
        - ```sh mover_tilt_down.sh```
        - ```sh mover_tilt_up.sh```
        - ```sh mover_zoom_in.sh```
        - ```sh mover_zoom_out.sh```
    - windows
        - ```camera_info.bat```
        - ```stream.bat```
        - ```mover_pan_left.bat```
        - ```mover_pan_right.bat```
        - ```mover_tilt_down.bat```
        - ```mover_tilt_up.bat```
        - ```mover_zoom_in.bat```
        - ```mover_zoom_out.bat```

2)	One-Shot learning 
    
    Пример использования (sh/bat скрипты):
    - обучение
        - ```sh train_osl.sh``` 
        -  ```train_osl.bat```
    -  инференс
        - 	```sh infer_osl.sh```
        - 	```infer_osl.bat```

    Набор данных: osl_faces
    
    Путь к набору данных (train): https://api.blackhole.ai.innopolis.university/public-datasets/testing/faces/train.zip

    Путь к набору данных (test): https://api.blackhole.ai.innopolis.university/public-datasets/testing/faces/test.zip

    Веса предобученной модели: https://api.blackhole.ai.innopolis.university//pretrained/one_shot_learning/epoch107.ckpt

3)	метод комплексирования данных сенсоров видимого диапазона света с данными, полученными от 2D- или 3D-сенсоров иной природы
    
    Пример использования (sh/bat скрипты):
    - ```sh infer_complexing_data.sh``` 
    -  ```infer_complexing_data.bat```

4)	алгоритм, основанный на рекуррентных нейронных сетях (lstm)

    Пример использования (sh/bat скрипты):
    - обучение
        - ```sh train_lstm.sh``` 
        -  ```train_lstm.bat```
    -  инференс
        - 	```sh infer_lstm.sh```
        - 	```infer_lstm.bat```

    Набор данных: ecg
    
    Путь к набору данных (train): https://api.blackhole.ai.innopolis.university/public-datasets/ECG/train.zip

    Путь к набору данных (test): https://api.blackhole.ai.innopolis.university/public-datasets/ECG/test.zip

    Веса предобученной модели: https://api.blackhole.ai.innopolis.university/pretrained/ecg_lstm.pt

5)	классификация

    Пример использования (sh/bat скрипты):
    - обучение
        - ```sh train_classification.sh``` 
        -  ```train_classification.bat```
    -  инференс
        - 	```sh infer_classification.sh```
        - 	```infer_classification.bat```
    
    Набор данных: mnist
    
    Путь к набору данных (train): https://api.blackhole.ai.innopolis.university/public-datasets/mnist/train.zip
    
    Путь к набору данных (test): https://api.blackhole.ai.innopolis.university/public-datasets/mnist/test.zip
    
    Веса предобученной модели: https://api.blackhole.ai.innopolis.university/pretrained/mnist_classification.pt.ckpt

6)	регрессия

    Пример использования (sh/bat скрипты):
    - обучение
        - ```sh train_linear_regression.sh``` 
        -  ```train_linear_regression.bat```
    -  инференс
        - 	```sh infer_linear_regression.sh```
        - 	```infer_linear_regression.bat```

    Набор данных: house_prices
    
    Путь к набору данных (train): https://api.blackhole.ai.innopolis.university/public-datasets/house_prices/train.zip

    Путь к набору данных (test): https://api.blackhole.ai.innopolis.university/public-datasets/house_prices/test.zip

    Веса предобученной модели: https://api.blackhole.ai.innopolis.university/pretrained/house_prices_lin_reg.pickle

7)	кластеризация

    Пример использования (sh/bat скрипты):
    - обучение
        - ```sh train_clustering.sh``` 
        -  ```train_clustering.bat```
    -  инференс
        - 	```sh infer_clustering.sh```
        - 	```infer_clustering.bat```

    Набор данных: credit_cards

    Путь к набору данных (train): https://api.blackhole.ai.innopolis.university/public-datasets/credit_cards/train.zip

    Путь к набору данных (test): https://api.blackhole.ai.innopolis.university/public-datasets/credit_cards/test.zip

    Веса предобученной модели: https://api.blackhole.ai.innopolis.university/pretrained/credit_cards_kmeans.pickle
