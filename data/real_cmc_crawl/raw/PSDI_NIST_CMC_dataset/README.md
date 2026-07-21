# Critical Micelle Concentrations of Aqueous Surfactant Systems

**Last updated:** 18-11-2024.

## Author Information
<meta charset="utf-8"><b style="font-weight:normal;" id="psdi-table"><div dir="ltr" style="margin-left:0pt;" align="left">
Author Name | Institute | Email | ORCID 
-- | -- | -- | --
Samuel Munday | Data Revival | samuel@data-revival.com | [https://orcid.org/0009-0009-9897-333X](https://orcid.org/0000-0001-5404-6934)
Ashley Unitt | Data Revival | ashley@data-revival.com | [https://orcid.org/0009-0003-9952-3468](https://orcid.org/0009-0007-0037-0035)
</div></b>

## General Information

This National Standards Reference Data System (NSRDS) book is a compilation of Critical Micelle Concentration's (CMC's) that have been manually collated, organised and evaluated from the literature from 1926 to 1966. In addition, over 800 values from graphs or implied in experimental data have been added to this dataset. In all, there are over 5000 entris based on 333 references dealing with 720 compounds. Where possible, the authors have included temperature, additives, method of determination and the literature source for a given CMC value.

This book, created in 1966, is not suitable for digital discovery due to it's unstrcutured format, therefore this repo contains the data extracted from the tables present in the book into a structured digital format. There are 2 files. The NIST Table of CMC values file represents a faithful digital copy (as faithful as possible) of the tables in the physical book. The CMC compounds file attempts to generate SMILES strings for each individual compounds.

The digital copy of this book can be found at: https://doi.org/10.6028/NBS.NSRDS.36, and there is also a local copy of the PDF in this repository. 

## Files 

Both the original xslx files and the formatted CSV files are included in this repo. If there are formatting issues in the CSV, please refer to the xlsx version. 

### NIST_Table_of_Critical_Micelle_Concentration_Values

Contains, as far as possible, a faithful digital representation of the orignal book. 

Headings are:

| Compound No. | Mol. Wgt. | Name | Additives | Temperature | CMC | Qual. Mat. Meth. | Method | Authors | Reference | Source | Evaluation | CMC Value 1 | CMC Units 1 | CMC Value 2 | CMC Units 2 | CMC Value 3 | CMC Units 3 |
|-------------|------------|------|-----------|-------------|-----|------------------|---------|---------|------------|---------|------------|--------------|--------------|--------------|--------------|--------------|--------------|
|             |            |      |           |             |     |                  |         |         |            |         |            |              |              |              |              |              |              |

Column descriptions:
- Compound No.: Unique identifier for each compound
- Mol. Wgt.: Molecular weight as reported in the original book
- Name: Chemical name as given in the original book (None IUPAC)
- Additives: Any additional substances present in the system
- Temperature: Temperature at which the CMC was measured
- CMC: Critical Micelle Concentration as reported in the original text
- Qual. Mat. Meth.: Qualitative material method
- Method: Experimental method used to determine the CMC
- Authors: Authors of the original research
- Reference: Original publication reference
- Source: Source of the data in the original book
- Evaluation: Quality assessment of the data
- CMC Value 1,2,3: Numerical CMC values split from original entries
- CMC Units 1,2,3: Units corresponding to each CMC value

### CMC Compounds

This file contains our attempt at generating SMILES strings for each compound. An old none-IUPAC naming convention is used in the book and so every attempt has been made to generate a valid modern name for conversion to SMILES. This is not 100% accurate, and a column is given to detail which method was used to generate the name/SMILE combo.

The general method was to use Llama 3.1 70b Large Language Model (LLM) from Meta, to make a best guess at what chemical species could be present from the original name. These were then assigned molecular weights, and the combination molecular weight was compared to the molecular weight presented in the CMC book. 

Headings are: 

| Compound No. | Mol. Wgt. | Name | IUPAC | SMILES | Lookup wgt. | Method |
|--------------|-----------|------|-------|--------|-------------|---------|
|              |           |      |       |        |             |         |

Column descriptions:
- Compound No.: Unique identifier matching the NIST table
- Mol. Wgt.: Molecular weight as reported in the original book
- Name: Chemical name as given in the original book
- IUPAC: Modern IUPAC name for the compound
- SMILES: Generated SMILES string representation of the molecule
- Lookup wgt.: Calculated molecular weight from the SMILES
- Method: Approach used to generate the IUPAC name and SMILES

## Known Limitations 

These data files are produced using both AI computer vision models and large language models. These make statistically derived best guesses for the information under review, and as such may make mistakes. Every care has been taken to manually review the data, however there may still exist inaccuracies. 

## License 



