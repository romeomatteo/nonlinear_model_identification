# mias_nonlinear_identification

With reference to the heat exchanger data:
  -ftp://ftp.esat.kuleuven.be/pub/SISTA/data/process_industry/exchanger.dat.gz
  -ftp://ftp.esat.kuleuven.be/pub/SISTA/data/process_industry/exchanger.txt
  (see also the process description in the cited paper)

Performed a nonlinear model identification using both the FROE method with polynomial NARX
models and feedforward neural networks. Dataset divided in identification (first 3000 data) and
validation data (the remaining 1000 data). Result compared with different model complexity assumptions.

Report available [here](report.docx)


References:
S. Bittanti and L. Piroddi, “Nonlinear identification and control of a heat exchanger: a neural
network approach”, Journal of the Franklin Institute, vol. 334B, pp. 135–153, 1997. 
