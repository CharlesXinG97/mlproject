import os
import numpy as np

data_dir = "RNA_trainset"
protein = ['AGO2', 'AGO1', 'AGO3', 'ALKBH5', 'AUF1', 'C17ORF85', 
        'C22ORF28', 'CAPRIN1', 'DGCR8', 'EIF4A3', 'EWSR1', 
        'FMRP', 'FOX2', 'FUS', 'FXR1', 'FXR2', 'HNRNPC', 'HUR', 
        'IGF2BP1', 'IGF2BP2', 'IGF2BP3', 'LIN28A', 'LIN28B', 
        'METTL3', 'MOV10', 'PTB', 'PUM2', 'QKI', 'SFRS1', 
        'TAF15', 'TDP43', 'TIA1', 'TIAL1', 'TNRC6', 'U2AF65', 
        'WTAP', 'ZC3H7B']

def load_raw_data(prot):
    """
    Load raw data from training dataset.

    Parameters:
    ----------
      prot: (string) the name of the protein.

    Returns:
    -------
      dna_seq: (list) a list of strings.
      binding: (list) a list of bool indicating whether the
               dna binds to the corresponding protein.
    """
    dna_seq = []
    binding = []
    data_file = os.path.join(data_dir, prot, 'train')
    with open(data_file, 'r') as f:
        datas = f.readlines()
        dna_seq.extend(data.split()[0] for data in datas)
        binding.extend(eval(data.split()[1]) for data in datas)
    return dna_seq, binding
         

DNA_seq, binding = load_raw_data('AGO2')
print(DNA_seq[0])
print(binding[0])
