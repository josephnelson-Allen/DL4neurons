# DL4neurons

## Running BBP models:

Compile modfiles: `$ nrnivmodl modfiles/*.mod` (all the ones you need are included in this repo)

```
$ python run.py --model BBP --m-type L5_TTPC1 --e-type cADpyr --cell-i 0 --stim-multiplier 2.0 --num 10 --outfile test.h5 --debug
```

This command should generate 10 traces and put them in the file test.h5

Also consider playing around with the `--stim-multiplier` option to run.py (I found a multiplier of 2.0 worked well-ish for the L5_TTPC1 cell), and the `--cell-i <integer>` option which allows you to select up to 5 different morphologies (`--cell-i 0` through `4`)).

If you want to run on cells other than L5_TTPC1, see below

#### Optional: Obtain cell models

The repo contains 5 example cells of m-type L5_TTPC1, e-type cADpyr. If you don't do the following optional steps, you will only be able to run with these 5 cells.

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

Then, add a list of parameters for the e-type such as the ones at [line 100 of models.py](https://github.com/VBaratham/DL4neurons/blob/master/models.py#L100)

Now you can use any m-type and e-type in the BBP model. A programmatically-accessible list of all m_types and e_types can be found in cells.json
