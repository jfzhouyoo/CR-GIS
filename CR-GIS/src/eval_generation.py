import pickle
from util.metrics import bleu,distinct,knowledge_hit,goal_hit, goal_hit_fuzz
import numpy as np
from util.utils import get_logger

def evaluate_generation(res_file, hyps_file, dataset, logger):

    with open(res_file,encoding="utf-8") as f: 
        refs = [result.split(" ") for result in f.readlines()]
    with open(hyps_file,encoding="utf-8") as f:
        hyps = [result.split(" ") for result in f.readlines()]

    report_message = []

    avg_len = np.average([len(s) for s in hyps])
    report_message.append("Avg_Len-{:.3f}".format(avg_len))

    bleu_1, bleu_2 = bleu(hyps, refs)
    report_message.append("Bleu-{:.4f}/{:.4f}".format(bleu_1, bleu_2))

    intra_dist1, intra_dist2, inter_dist1, inter_dist2 = distinct(hyps)
    report_message.append("Inter_Dist-{:.4f}/{:.4f}".format(inter_dist1, inter_dist2))
    report_message.append("Intra_Dist-{:.4f}/{:.4f}".format(intra_dist1, intra_dist2))

    print(str(report_message))

    logger.info("\n".join(report_message))

if __name__=='__main__':
    dataset = "tgredial"
    res_file =  "../models/responses.txt"
    hyps_file = "../models/results.txt" 
    logger = get_logger("../logs/eval_generation.log")
    evaluate_generation(res_file, hyps_file, dataset, logger=logger)