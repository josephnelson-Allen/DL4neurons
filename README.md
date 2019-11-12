# DL4neurons

Quickstart:

`$ nrnivmodl modfiles/*.mod`

`$ python run.py --help`

## Running BBP models

Compile modfiles: `$ nrnivmodl modfiles/*.mod` (all the ones you need are included in this repo)

Obtain cell models from https://bbp.epfl.ch/nmc-portal/web/guest/downloads

They will arrive as a zip file called models.py, which you should save into a directory called hoc_templates alongside run.py

Then enter hoc_templates and unzip models.py, and all the individual cells:

```
$ cd hoc_templates
$ unzip models.zip
$ rm models.zip
$ ls
L1_DAC_cNAC187_1.zip	L1_DAC_cNAC187_2.zip	L1_DAC_cNAC187_3.zip	L1_DAC_cNAC187_4.zip	L1_DAC_cNAC187_5.zip [...]
$ unzip '*.zip' # quotes are necessary!!
$ ls
L1_DAC_cNAC187_1	L1_DAC_cNAC187_2	L1_DAC_cNAC187_3	L1_DAC_cNAC187_4	L1_DAC_cNAC187_5
L1_DAC_cNAC187_1.zip	L1_DAC_cNAC187_2.zip	L1_DAC_cNAC187_3.zip	L1_DAC_cNAC187_4.zip	L1_DAC_cNAC187_5.zip
$ rm *.zip
$ ls
L1_DAC_cNAC187_1	L1_DAC_cNAC187_2	L1_DAC_cNAC187_3	L1_DAC_cNAC187_4	L1_DAC_cNAC187_5
```

Currently you have to run a separate command to create the output file (this helps when generating data from multiple threads):

```
$ python run.py --model BBP --m-type L5_TTPC1 --e-type cADpyr --num 10 --outfile test.h5 --debug --create
$ python run.py --model BBP --m-type L5_TTPC1 --e-type cADpyr --num 10 --outfile test.h5 --debug
```

These commands should generate 10 test traces and put them in the file test.h5
