import warnings
warnings.filterwarnings('ignore')

import re
import os
import sys
import time
import json
import argparse
import pandas as pd
import os.path as osp
from sklearn.metrics import classification_report

from utils import Logger
from annotator import AnalogyAutoAnnotator

current_path = os.path.dirname(__file__)


def main(all_analogies, args, dirname):
    annotator = AnalogyAutoAnnotator(
        llm_name=args.llm_model, 
        api_key=args.api_key,
        base_url=args.base_url,
        temperature=args.temperature, 
        verbose=args.verbose,
        e2e=args.e2e,
        en_prompt=args.en_prompt,
        modify_dsrt=True,
        summarize_by_llm=False)
    num_all_pairs = len(all_analogies)

    result_list = []
    pred_label_pool = []
    ground_truth_pool = []
    false_ana, surface_ana, appear_ana, anomaly_ana, near_ana, far_ana = 0, 0, 0, 0, 0, 0
    false_ana_c, surface_ana_c, appear_ana_c, anomaly_ana_c, near_ana_c, far_ana_c = 0, 0, 0, 0, 0, 0
    prompt_tokens, completion_tokens, total_tokens = 0, 0, 0
    acc_count = 0

    idx = 0
    start_time = time.time()
    for sample_data in all_analogies:
        try:
            ground_truth = sample_data['ground_truth']
            base, target = sample_data['base'], sample_data['target']
            actual_pred_type, reason_process, summarize_json = annotator.run(sample_data)
            ground_truth_pool.append(ground_truth)
            pred_label_pool.append(actual_pred_type)

            result_list.append({
                "id": f"{args.data_file}_{(idx+1):04}",
                "story_a": base,
                "story_b": target,
                "ground_truth": ground_truth,
                "predict_label": actual_pred_type,
                "reasoning_process": reason_process,
                "rectify_reasoning": summarize_json
            })

            if actual_pred_type == ground_truth:
                acc_count += 1
            if actual_pred_type == 'False Analogy':
                false_ana += 1
                if actual_pred_type == ground_truth:
                    false_ana_c += 1
            elif actual_pred_type == 'Surface Similar':
                surface_ana += 1
                if actual_pred_type == ground_truth:
                    surface_ana_c += 1
            elif actual_pred_type == 'Mere Appearance':
                appear_ana += 1
                if actual_pred_type == ground_truth:
                    appear_ana_c += 1
            elif actual_pred_type == 'Anomaly':
                anomaly_ana += 1
                if actual_pred_type == ground_truth:
                    anomaly_ana_c += 1
            elif actual_pred_type == 'Literally Similar':
                near_ana += 1
                if actual_pred_type == ground_truth:
                    near_ana_c += 1
            elif actual_pred_type == 'True Analogy':
                far_ana += 1
                if actual_pred_type == ground_truth:
                    far_ana_c += 1

            print(f"story_a: {base}\nstory_b: {target}\n{idx+1} | {num_all_pairs} finished.  Predict: {actual_pred_type} | Actual: {ground_truth} \n\n")
        except KeyboardInterrupt:
            break
        except Exception:
            pass
        idx += 1
    end_time = time.time()

    prompt_tokens += annotator.tokens_usages["prompt_tokens"]
    completion_tokens += annotator.tokens_usages["completion_tokens"]
    total_tokens += annotator.tokens_usages["total_tokens"]

    result_stats_df = pd.DataFrame([{"prompt_tokens": prompt_tokens, 
                                     "completion_tokens": completion_tokens, 
                                     "total_tokens": total_tokens,
                                     "literally_similar_count": f"{near_ana}({near_ana_c})",
                                     "true_analogy_count": f"{far_ana}({far_ana_c})",
                                     "false_analogy_count": f"{false_ana}({false_ana_c})",
                                     "surface_similar_count": f"{surface_ana}({surface_ana_c})",
                                     "mere_appearance_count": f"{appear_ana}({appear_ana_c})",
                                     "anomaly_count": f"{anomaly_ana}({anomaly_ana_c})",
                                     "accuracy": f"{acc_count/num_all_pairs:.2f}",
                                     "time_cost": f"{end_time-start_time:.2f}s"}])

    if args.save_results:
        result_stats_df.to_csv(f"{dirname}/result_stats_{args.llm_model}_{args.data_file}.csv", index=False)
    
        json_file = f"{dirname}/eval_analogy_{args.llm_model}_{args.data_file}.json"
        with open(json_file, 'w', encoding='utf-8') as file:
            json.dump(result_list, file, ensure_ascii=False, indent=2)

    print(f"Total pairs: {num_all_pairs}")
    print(f"Accuracy: {acc_count/num_all_pairs:.2f}")
    print(f"Time cost: {end_time-start_time:.2f}s")
    print(f"Result Stats:\n{result_stats_df.T.to_markdown()}")
    print("-"*50)
    print(classification_report(y_true=ground_truth_pool, y_pred=pred_label_pool))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--llm_model', '-LM', type=str, default='gpt-4o', help='Language model name')
    argparser.add_argument('--auxiliary_llm_model', '-AM', type=str, default='gpt-4o', help='Language model name')
    argparser.add_argument('--temperature', '-T', type=float, default=0.01, help='Temperature for GPT completions')
    argparser.add_argument('--max_tokens', '-K', type=int, default=2048, help='Max tokens for GPT completions')
    argparser.add_argument('--output_dir', '-O', type=str, default="output", help='Result save path')
    argparser.add_argument('--data_file', '-D', type=str, default="cognitive_psychology_en", help='Test dataset file name')
    argparser.add_argument('--en_prompt', '-EP', type=bool, default=True, help='Use english prompt to instruct LLM')
    argparser.add_argument('--verbose', '-V', type=bool, default=True, help='print log')
    argparser.add_argument('--save_results', '-SR', type=bool, default=True, help='save results')
    argparser.add_argument('--save_logging', '-SL', type=bool, default=True, help='save log')
    argparser.add_argument('--e2e', '-E2E', type=bool, default=False, help='End to End(one-step) to verify')
    argparser.add_argument('--api_key', type=str, default=None, help='LLM api key')
    argparser.add_argument('--base_url', type=str, default=None, help='LLM base_url')
    args = argparser.parse_args()

    with open(f"{current_path}/data/{args.data_file}.json", "r") as file:
        all_analogies = json.load(file)

    for root, dirs, files in os.walk(f'{current_path}/{args.output_dir}', topdown=False):
        for dir_name in dirs:
            folder_path = os.path.join(root, dir_name)
            if not os.listdir(folder_path):
                os.rmdir(folder_path)

    date_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if args.save_logging:
        dirname = osp.join(f'{current_path}/{args.output_dir}', f"{date_str}_{args.llm_model}")
        if not osp.exists(dirname):
            os.mkdir(dirname)
        sys.stdout = Logger(osp.join(dirname, f'log_{args.llm_model}_{args.data_file}_logger.txt'))
    print(args)

    main(all_analogies, args, dirname)