Для использования Фреймворка для поддержки принятия решений в поиске материалов с заданными свойствами необходимо выполнить следующие шаги:
1)	Склонировать фреймворк;
git clone https://github.com/InnopolisUni/innofw.git
2)	Установить пакеты poetry;
poetry install
3)	Для инициализации модели предобученными весами указать путь к весам в параметре ckpt_path эксперимента, либо в shell/batch скрипта;
4)	Путь к наборам данных указан в параметре source конфигурационных файлов datasets (набор данных загрузится автоматически при запуске скрипта);
5)	Запустить алгоритмы посредством shell/batch скриптов.
Команды с использованием shell скриптов приведенные ниже должны быть использованы на ОС Линукс.
Команды с использованием batch скриптов приведенные ниже должны быть использованы на ОС Windows.
Список алгоритмов:

    1)	Реализация возможности обеспечить решение задачи «Прогнозирование свойств материалов на основании их структуры»

        Пример использования (sh/bat скрипты):
        -	train_QSAR_test.sh
        -	train_QSAR_test.bat
        -	infer_QSAR_test.sh
        -	infer_QSAR_test.bat
        
        Набор данных: qm9
        
        Путь к набору данных (train): https://api.blackhole.ai.innopolis.university/public-datasets/qm9/train.zip
        
        Путь к набору данных (test): https://api.blackhole.ai.innopolis.university/public-datasets/qm9/test.zip
        
        Веса предобученной модели: https://api.blackhole.ai.innopolis.university/pretrained/catboost_regression_qm9.pickle

    2)	Реализация возможности обеспечить решение задачи «Прогнозирование технологических параметров производства»

        Пример использования (sh/bat скрипты):
        - train_Catboost_test.sh
        -	train_Catboost_test.bat
        -	infer_Catboost_test.sh
        -	infer_Catboost_test.bat

        Набор данных: regression_industry_data

        Путь к набору данных (train): https://api.blackhole.ai.innopolis.university/public-datasets/industry_data/train.zip

        Путь к набору данных (test): https://api.blackhole.ai.innopolis.university/public-datasets/industry_data/test.zip

        Веса предобученной модели: https://api.blackhole.ai.innopolis.university/pretrained/catboost_regression_industry_data.pickle

    3)	Реализация возможности обеспечить решение задачи «Анализ данных из структурированных и неструктурированных источников»

        Пример использования (sh/bat скрипты):
        -	train_biobert_test.sh
        -	train_biobert_test.bat
        -	infer_biobert_test.sh
        -	infer_biobert_test.bat

        Набор данных: token_classification_drugprot

        Путь к набору данных (train): https://api.blackhole.ai.innopolis.university/public-datasets/drugprot/train.zip

        Путь к набору данных (test): https://api.blackhole.ai.innopolis.university/public-datasets/drugprot/test.zip

        Веса предобученной модели: https://api.blackhole.ai.innopolis.university/pretrained/biobert_ner_drugprot.ckpt

    4)	Метод проверки физических дескрипторов
    
        Пример использования (sh/bat скрипты):
        -	train_QSAR_test.sh
        -	train_QSAR_test.bat
        -	infer_QSAR_test.sh
        -   infer_QSAR_test.bat

        Набор данных: qm9

        Путь к набору данных (train): https://api.blackhole.ai.innopolis.university/public-datasets/qm9/train.zip

        Путь к набору данных (test): https://api.blackhole.ai.innopolis.university/public-datasets/qm9/test.zip

        Веса предобученной модели: https://api.blackhole.ai.innopolis.university/pretrained/catboost_regression_qm9.pickle

    5)	Метрик и методов оценки неопределенности данных и неопределенности моделей
    
        Пример использования (sh/bat скрипты):
        -	train_Catboost_uncertainty_test.sh
        -	train_Catboost_uncertainty_test.bat
        -	infer_Catboost_uncertainty_test.sh
        -	infer_Catboost_uncertainty_test.bat
    
        Набор данных: qm9
    
        Путь к набору данных (train): https://api.blackhole.ai.innopolis.university/public-datasets/qm9/train.zip
    
        Путь к набору данных (test): https://api.blackhole.ai.innopolis.university/public-datasets/qm9/test.zip
    
        Веса предобученной модели: https://api.blackhole.ai.innopolis.university/pretrained/catboost_active_learning.pickle

    6)	Алгоритмы дизайна эксперимента, поиска материалов с заданными свойствами
    
        Пример использования (sh/bat скрипты):
        -	train_QSAR_test.sh
        -	train_QSAR_test.bat
        -	infer_QSAR_test.sh
        -	infer_QSAR_test.bat

        Набор данных: qm9

        Путь к набору данных (train): https://api.blackhole.ai.innopolis.university/public-datasets/qm9/train.zip

        Путь к набору данных (test): https://api.blackhole.ai.innopolis.university/public-datasets/qm9/test.zip

        Веса предобученной модели: https://api.blackhole.ai.innopolis.university/pretrained/catboost_regression_qm9.pickle

    7)	Aвтокодировщики (автоэнкодеры) данных

        Пример использования (sh/bat скрипты):
        -	train_VAE_test.sh
        -	train_VAE _test.bat
        -	infer_VAE _test.sh
        -	infer_VAE _test.bat

        Набор данных: qm9

        Путь к набору данных (train): https://api.blackhole.ai.innopolis.university/public-datasets/qm9/train.zip

        Путь к набору данных (test): https://api.blackhole.ai.innopolis.university/public-datasets/qm9/test.zip

        Веса предобученной модели: https://api.blackhole.ai.innopolis.university/pretrained/chem_vae.ckpt

    8)	Алгоритм для решения прямой и обратной задачи поиска новых материалов

        Пример использования (sh/bat скрипты):
        -	train_VAE_test.sh
        -	train_VAE _test.bat
        -	infer_VAE _test.sh
        -	infer_VAE _test.bat
        -	infer_VAE_reverce_test.sh
        -	infer_VAE_reverce_test.bat

        Набор данных: qm9

        Путь к набору данных (train): https://api.blackhole.ai.innopolis.university/public-datasets/qm9/train.zip

        Путь к набору данных (test): https://api.blackhole.ai.innopolis.university/public-datasets/qm9/test.zip

        Веса предобученной модели: https://api.blackhole.ai.innopolis.university/pretrained/chem_vae.ckpt

    9)	Алгоритмы с активным обучением, применяемые к широкому спектру систем, включая как кристаллические, так и молекулярные структуры

        Пример использования (sh/bat скрипты):
        -	train_QSAR_AL_test.sh
        -	train_QSAR_AL _test.bat
        -	infer_QSAR_AL _test.sh
        -	infer_QSAR_AL _test.bat

        Набор данных: qm9

        Путь к набору данных (train): https://api.blackhole.ai.innopolis.university/public-datasets/qm9/train.zip

        Путь к набору данных (test): https://api.blackhole.ai.innopolis.university/public-datasets/qm9/test.zip

        Веса предобученной модели: https://api.blackhole.ai.innopolis.university/pretrained/catboost_active_learning.pickle

    10)	Алгоритмы для решения задач квантовой механики

        Пример использования (sh/bat скрипты):
        -	train_QSAR_test.sh
        -	train_QSAR_test.bat
        -	infer_QSAR_test.sh
        -	infer_QSAR_test.bat

        Набор данных: qm9

        Путь к набору данных (train): https://api.blackhole.ai.innopolis.university/public-datasets/qm9/train.zip

        Путь к набору данных (test): https://api.blackhole.ai.innopolis.university/public-datasets/qm9/test.zip

        Веса предобученной модели: https://api.blackhole.ai.innopolis.university/pretrained/catboost_regression_qm9.pickle
