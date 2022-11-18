## Инструкция по установке фреймворка на ASTRA Linux

1. скачать Miniconda3 (ссылка: https://docs.conda.io/en/latest/miniconda.html)

2. перейти в каталог, куда был скачан shell скрипт

3. выполнить команду по запуску скрипта: bash Miniconda3-latest-Linux-x86_64.sh

4. перезагрузить терминал

5. убедиться в установленной версии: conda list

6. убедиться в установленной версии python 3.9: python –version

7. обновить репозиторий: sudo apt-get update

8. установить систему контроля версий Git: sudo apt-get install git

9. клонировать фреймворк: git clone https://github.com/InnopolisUni/innofw

10. установить poetry: conda install -c conda-forge poetry

11. перезагрузить терминал

12. перейти в папку, куда установлен фреймворк: cd innofw

13. выполнить команду по активации среды conda: conda activate py38

14. выполнить команду по активации виртуальной среды (активировать виртуальную среду для работы с poetry необходимо каждую сессию): poetry shell

15. выполнить команду: poetry lock --no-update

16. установить пакеты, указанные в файле poetry.lock командой: poetry install

17. установить CUDA с официального сайта (https://developer.nvidia.com/cuda-downloads

18. если необходимо изменить состав или версии пакетов, включаемых во Фреймворк, то необходимо внести изменения в файл pyproject.toml в папке, куда был склонирован проект. А также удалить файл poetry.lock в той же папке, если он там присутствует
