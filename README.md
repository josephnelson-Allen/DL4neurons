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

Obtain cell models from https://bbp.epfl.ch/nmc-portal/web/guest/downloads (Use the link where it says "The complete set of neuron models is available here")

They will arrive as a zip file called hoc_combos_syn.1_0_10.allzips.tar, which you should save into a directory called hoc_templates alongside run.py

Then enter hoc_templates and untar/unzip everything:

```
$ ls
run.py  models.py  hoc_templates  [...]  hoc_combos_syn.1_0_10.allzips.tar
$ tar -xvf hoc_combos_syn.1_0_10.allzips.tar --directory hoc_templates
$ cd hoc_templates
$ unzip 'hoc_combos_syn.1_0_10.allzips/*.zip' # quotes are necessary!!
$ ls
L1_DAC_cNAC187_1	L1_DAC_cNAC187_2	L1_DAC_cNAC187_3	L1_DAC_cNAC187_4	L1_DAC_cNAC187_5 [...] hoc_combos_syn.1_0_10.allzips

## Cleanup
$ rm -r hoc_combos_syn.1_0_10.allzips
$ cd ..
$ rm hoc_combos_syn.1_0_10.allzips.tar
```

You should have the following structure:

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

Now you can use any m-type and e-type in the BBP model. 

A programmatically-accessible list of all m_types and e_types can be found in cells.json

## Some use cases

### Running the same parameter set with different stimuli

First, generate the parameter set csv:

```
python run.py --model BBP --m-type L5_TTPC1 --e-type cADpyr --num 100 --param-file params.csv --create-params
```
(or you can create it by yourself, by hand, or whatever method you like. Run this code for an example of what it should look like)

You must put the stimulus into a csv file. See the "stims" directory in this repo for examples.

Then pass this params file along with the stimulus file to run.py:

```
python run.py --model BBP --m-type L5_TTPC1 --e-type cADpyr --outfile results_stim1.h5 --param-file params.csv --stim-file stims/chaotic_1.csv
python run.py --model BBP --m-type L5_TTPC1 --e-type cADpyr --outfile results_stim2.h5 --param-file params.csv --stim-file stims/some_other_stim.csv
```
