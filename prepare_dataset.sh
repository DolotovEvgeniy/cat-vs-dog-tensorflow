wget https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip
unzip kagglecatsanddogs_3367a.zip -d data-tmp
mkdir cat-vs-dog
mv data-tmp/PetImages/Dog cat-vs-dog/dog
mv data-tmp/PetImages/Cat cat-vs-dog/cat
rm -r kagglecatsanddogs_3367a.zip data-tmp

python clean_dataset.py

python train_test_valid_split.py
