Для использования Фреймворка для поддержки принятия врачебных решений в необходимо выполнить следующие шаги:
1.	Склонировать фреймворк;
    
    ```git clone https://github.com/InnopolisUni/innofw.git```

2.	Установить пакеты poetry;

    ```poetry install```

3.	Для инициализации модели предобученными весами указать путь к весам в параметре ckpt_path эксперимента, либо в shell/batch скрипта;

4.	Путь к наборам данных указан в параметре source конфигурационных файлов datasets (набор данных загрузится автоматически при запуске скрипта);
5.	Запустить алгоритмы посредством shell/batch скриптов.
Команды с использованием shell скриптов приведенные ниже должны быть использованы на ОС Линукс.
Команды с использованием batch скриптов приведенные ниже должны быть использованы на ОС Windows.
    1.	Метод предобработки рентгенологических снимков и данных КТ(метод повышения контраста пораженных тканей)

        Пример использования (sh/bat скрипты):
        -	run_kernel_trick.sh
        -	run_ribs_supression.sh

        -	run_kernel_trick.bat
        -	run_ribs_supression.bat


        

    2.	Метода синхронизации с медицинским отраслевым стандартом DICOM

        Пример использования (sh/bat скрипты):
        -	infer_brain_segmentation.sh
        -	infer_brain_segmentation.bat

    3.	Локализации патологий на медицинских изображениях (рентгенологических и КТ снимках)

        Пример использования (sh/bat скрипты):
        -	обучение
            -	train_brain_segmentation.sh
            -	train_lung_detection.sh
            -	train_brain_segmentation.bat
            -	train_lung_detection.bat
        -	инференс
            -	infer_brain_segmentation.sh
            -	infer_lung_detection.sh
            -	infer_brain_segmentation.bat
            -	infer_lung_detection.bat

        Набор данных: brain_ct

        Путь к набору данных (train): https://api.blackhole.ai.innopolis.university/public-datasets/stroke/train.zip

        Путь к набору данных (test): https://api.blackhole.ai.innopolis.university/public-datasets/stroke/test.zip

        Веса предобученной модели: https://api.blackhole.ai.innopolis.university/pretrained/segmentation_unet_brain.pt

        Набор данных: lung_detection

        Путь к набору данных (train): https://api.blackhole.ai.innopolis.university/public-datasets/lungs_detection/train.zip

        Путь к набору данных (test): https://api.blackhole.ai.innopolis.university/public-datasets/lungs_detection/test.zip

        Веса предобученной модели: https://api.blackhole.ai.innopolis.university/pretrained/lung_detection.pt

    4.	Cегментации медицинских изображений (КТ снимков)

        Пример использования (sh/bat скрипты):
        -	обучение
            -	train_brain_segmentation.sh
            -	train_brain_segmentation.bat
        -	инференс
            -	infer_brain_segmentation.sh
            -	infer_brain_segmentation.bat

        Набор данных: brain_ct

        Путь к набору данных (train): https://api.blackhole.ai.innopolis.university/public-datasets/stroke/train.zip

        Путь к набору данных (test): https://api.blackhole.ai.innopolis.university/public-datasets/stroke/test.zip

        Веса предобученной модели: https://api.blackhole.ai.innopolis.university/pretrained/segmentation_unet_brain.pt

    5.	Метод трансформации медицинских изображений(Dicom в JPG)

        Пример использования (sh/bat скрипты):
        -	run_dicom_to_image.sh
        -	run_dicom_to_image.sh
