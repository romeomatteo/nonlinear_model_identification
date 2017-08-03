# mias_nonlinear_identification

With reference to the heat exchanger data:
  -ftp://ftp.esat.kuleuven.be/pub/SISTA/data/process_industry/exchanger.dat.gz
  -ftp://ftp.esat.kuleuven.be/pub/SISTA/data/process_industry/exchanger.txt
  (see also the process description in the cited paper)

perform a nonlinear model identification using both the FROE method with polynomial NARX
models and feedforward neural networks. Divide the dataset in identification (first 3000 data) and
validation data (the remaining 1000 data). Use the former only for model estimation and the latter
for model evaluation, both in terms of prediction and simulation accuracy. Compare the results with
different model complexity assumptions.


References:
S. Bittanti and L. Piroddi, “Nonlinear identification and control of a heat exchanger: a neural
network approach”, Journal of the Franklin Institute, vol. 334B, pp. 135–153, 1997. 
