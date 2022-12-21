Список алгоритмов:
1)	интеллектуальный анализ и мониторинг потоков данных, система сетевого видеонаблюдения для обеспечения видеоаналитики и работы с потоками данных (ONVIF)
    
    Пример использования (sh/bat скрипты):
    -	camera_info.sh
    -	stream.sh
    -	mover_pan_left.sh
    -	mover_pan_right.sh
    -	mover_tilt_down.sh
    -	mover_tilt_up.sh
    -	mover_zoom_in.sh
    -	mover_zoom_out.sh

2)	One-Shot learning 
    
    Пример использования (sh/bat скрипты):
    -	train_osl.sh
    -	infer_osl.sh

    Набор данных: osl_faces
    Путь к набору данных (train): https://api.blackhole.ai.innopolis.university/public-datasets/testing/faces/train.zip

    Путь к набору данных (test): https://api.blackhole.ai.innopolis.university/public-datasets/testing/faces/test.zip

    Веса предобученной модели: https://api.blackhole.ai.innopolis.university//pretrained/one_shot_learning/epoch107.ckpt

3)	метод комплексирования данных сенсоров видимого диапазона света с данными, полученными от 2D- или 3D-сенсоров иной природы
    
    Пример использования (sh/bat скрипты):
    - infer_complexing_data.sh

4)	алгоритм, основанный на рекуррентных нейронных сетях (lstm)

    Пример использования (sh/bat скрипты):
    -	train_lstm.sh
    -	infer_lstm.sh

    Набор данных: ecg
    Путь к набору данных (train): https://api.blackhole.ai.innopolis.university/public-datasets/ECG/train.zip

    Путь к набору данных (test): https://api.blackhole.ai.innopolis.university/public-datasets/ECG/test.zip

    Веса предобученной модели: https://api.blackhole.ai.innopolis.university/pretrained/ecg_lstm.pt

5)	классификация

    Пример использования (sh/bat скрипты):
    -	train_classification.sh
    -	infer_classification.sh
    
    Путь к набору данных (train): https://api.blackhole.ai.innopolis.university/public-datasets/mnist/train.zip
    
    Путь к набору данных (test): https://api.blackhole.ai.innopolis.university/public-datasets/mnist/test.zip
    
    Веса предобученной модели: https://api.blackhole.ai.innopolis.university/pretrained/mnist_classification.pt.ckpt

6)	регерссия

    Пример использования (sh/bat скрипты):
    -	train_linear_regression.sh
    -	infer_linear_regression.sh

    Путь к набору данных (train): https://api.blackhole.ai.innopolis.university/public-datasets/house_prices/train.zip

    Путь к набору данных (test): https://api.blackhole.ai.innopolis.university/public-datasets/house_prices/test.zip

    Веса предобученной модели: https://api.blackhole.ai.innopolis.university/pretrained/house_prices_lin_reg.pickle

7)	кластеризация

    Пример использования (sh/bat скрипты):
    -	train_clustering.sh
    -	infer_clustering.sh

    Путь к набору данных (train): https://api.blackhole.ai.innopolis.university/public-datasets/credit_cards/train.zip

    Путь к набору данных (test): https://api.blackhole.ai.innopolis.university/public-datasets/credit_cards/test.zip

    Веса предобученной модели: https://api.blackhole.ai.innopolis.university/pretrained/credit_cards_kmeans.pickle
