import os
import random
import numpy as np

DATA_DIR = "RNA_trainset"
PROTEIN = ['AGO2', 'AGO1', 'AGO3', 'ALKBH5', 'AUF1', 'C17ORF85', 
        'C22ORF28', 'CAPRIN1', 'DGCR8', 'EIF4A3', 'EWSR1', 
        'FMRP', 'FOX2', 'FUS', 'FXR1', 'FXR2', 'HNRNPC', 'HUR', 
        'IGF2BP1', 'IGF2BP2', 'IGF2BP3', 'LIN28A', 'LIN28B', 
        'METTL3', 'MOV10', 'PTB', 'PUM2', 'QKI', 'SFRS1', 
        'TAF15', 'TDP43', 'TIA1', 'TIAL1', 'TNRC6', 'U2AF65', 
        'WTAP', 'ZC3H7B']

def load_raw_data(prot, shuffle=True):
    """
    Load raw data from training dataset.

    Parameters:
    ----------
      prot: (string) the name of the protein.
      shuffle: (bool) if True shuffle the data.

    Returns:
    -------
      dna_seq: (list) a list of strings.
      binding: (list) a list of bool indicating whether the
               dna binds to the corresponding protein.
    """
    dna_seq = []
    binding = []
    data_file = os.path.join(DATA_DIR, prot, 'train')
    with open(data_file, 'r') as f:
        datas = f.readlines()
        if shuffle:
            random.shuffle(datas)
        dna_seq.extend(data.split()[0] for data in datas)
        binding.extend(eval(data.split()[1]) for data in datas)
    return dna_seq, binding
         
def dna_segmentation(dna_seq, seg=3):
    """
    """
    seg_dna = []
    i = 0
    while i < len(dna_seq):
        seg_dna.append(dna_seq[i:i+seg])
        i += seg
    return seg_dna

def roc(ground_truth, pred_result):
    """
    Calculate the roc value of given results and ground truth.
    Parameters:
    ----------
      ground_truth: (nd_array) an array of label ground truth.
      pred_result: (nd_array) an array of predicted label.
    Returns:
    -------
      roc: a point in roc space, the x-axis is the false positive rate
           while the y-axis is the true positive rate.
    """
    assert len(ground_truth)==len(pred_result)
    tp, fp, tn, fn = 1e-8, 1e-8, 1e-8, 1e-8
    for i in range(len(ground_truth)):
        if ground_truth[i][0] == 0 and pred_result[i][0] == 0:
            tp += 1
        elif ground_truth[i][0] == 0 and pred_result[i][0] == 1:
            fn += 1
        elif ground_truth[i][0] == 1 and pred_result[i][0] == 0:
            fp += 1
        elif ground_truth[i][0] == 1 and pred_result[i][0] == 1:
            tn += 1
    roc_tpr, roc_fpr = tp/(tp+fn), fp/(fp+tn)
    return (roc_fpr, roc_tpr)

# TODO: eliminate redundent data
# TODO: shuflle all data to test
