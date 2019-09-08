# Project for Biometría, UPV

In this project I've been working with face recognition using eigenfaces and fisherfaces with/without light normalization. 

- Report_Biometria_Olsen.pdf contains project report. 

- import_data.py imports data (not added to git) from ORL database/Yale database. 

- functions.py contains all necessary functions. 

- main.py contains code to get plots in report. 


All the functions are run from the main. They are listed with a comment of what they do. Data is not added to git, since data is images that requires a lot of space. To run code, data from ORL/Yale database must be downloaded and added to folder with structure /Images/s<#individual>/<#image>.pgm, where #individual = {1, ..., 40} and #image = {1, ..., 10} for ORL database, and /Images/d<#individual>/<#image>.pgm, where #individual = {1, ..., 10} and #image = {1, ..., 10} for Yale database. Ex: Images/s21/7..
