# DL4neurons

Quickstart:

`$ nrnivmodl modfiles/*.mod`

`$ python run.py --help`

## Running BBP models

Compile modfiles: `$ nrnivmodl modfiles/*.mod` (all the ones you need are included in this repo)

#### Optional: Obtain cell models

The repo contains 5 example cells of m-type L5_TTPC1, e-type cADpyr. If you don't do these optional steps, you will only be able to run with these 5 cells.

Obtain cell models from https://bbp.epfl.ch/nmc-portal/web/guest/downloads

They will arrive as a zip file called models.zip, which you should save into a directory called hoc_templates alongside run.py

Then enter hoc_templates and unzip models.zip, and all the individual cells zipped within:

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

Basically, you want the following structure:

```
DL4neurons/
   run.py
   models.py
   [...]
   hoc_templates/
       L1_DAC_cNAC187_1/
       L1_DAC_cNAC187_2/
       L1_DAC_cNAC187_3/
       [...]
       L6_UTPC_cADpyr231_1/
       L6_UTPC_cADpyr231_2/
```

## Run the simulations

Currently you have to run a separate command to create the output file (this helps when generating data from multiple threads):

```
$ python run.py --model BBP --m-type L5_TTPC1 --e-type cADpyr --num 10 --outfile test.h5 --debug --create
$ python run.py --model BBP --m-type L5_TTPC1 --e-type cADpyr --num 10 --outfile test.h5 --debug
```

These commands should generate 10 test traces and put them in the file test.h5

A programmatically-accessible list of all m_types and e_types can be found in cells.json

Also consider playing around with the `--stim-multiplier` option to run.py (I found a multiplier of 2.0 worked well-ish for the L5_TTPC1 cell), and the `--cell-i <integer>` option which allows you to select a morphology clone (currently cells.json only knows about 2 clones, so your only options are `--cell-i 0` (default) and `--cell-i 1`).
