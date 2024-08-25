import argparse
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="model")
parser.add_argument(
    "--decoder", type=str, help="decoding strategy")

args = parser.parse_known_args()[0]


input_file_name = f"generated_captions/{args.model}/HallusionBench_{args.decoder}_result.json"
save_json_path_vd = f"generated_captions/HallusionBench_{args.decoder}_vd_model.json"
load_json = True
model_output_entry = "model_prediction"
model_correctness_entry = "gpt4v_output_gpt_check"


if __name__ == "__main__":

    data_vd = []
    data_vs = []
    with open(input_file_name) as json_file:
        datas = json.load(json_file)

    for data in tqdm(datas):
        if data['category'] == 'VD':
            data_vd.append(data)
                      
    data_vd = evaluate_by_chatgpt(data_vd, model_output_entry, model_correctness_entry, load_json=load_json, save_json_path=save_json_path_vd)
    data_vd = check_same_by_chatgpt(data_vd, model_output_entry, load_json=load_json, save_json_path=save_json_path_vd)

    print("##### GPT Evaluate #####")

    data_vd = assign_correctness(data_vd, correctness_entry=model_correctness_entry)
    data = data_vd

    all_data = get_eval_all(data, model_correctness_entry)
    all_vd = get_eval_all(data_vd, model_correctness_entry)

    table1 = [["per question", "Total"], 
              ["VD", round(100 * all_vd["correct"]/all_vd["total"], 4)],
              ["Overall", round(100 * all_data["correct"]/all_data["total"], 4)]]
    tab1 = PrettyTable(table1[0])
    tab1.add_rows(table1[1:])

    q_acc_gpt = round(100 * all_data["correct"]/all_data["total"], 4)

    fig_all = get_eval_fig(data)
    fig_vd = get_eval_fig(data_vd)

    # image level 
    table2 = [["per figure", "Correct", "Wrong", "Score"], 
              ["VD", round(100 * fig_vd["correct"]/fig_vd["total"], 4), round(100 * fig_vd["inconsistent"]/fig_vd["total"], 4) + round(100 * fig_vd["wrong"]/fig_vd["total"], 4), round(fig_vd["score"], 4)],
              ["Overall", round(100 * fig_all["correct"]/fig_all["total"], 4), round(100 * fig_all["inconsistent"]/fig_all["total"], 4) + round(100 * fig_all["wrong"]/fig_all["total"], 4), round(fig_all["score"], 4)]]
    tab2 = PrettyTable(table2[0])
    tab2.add_rows(table2[1:])

    pair_acc_gpt = round(100 * all_data["correct"]/all_data["total"], 4)
    figure_acc_gpt = round(100 * fig_all["correct"]/fig_all["total"], 4)

    print("##### Figure Stats #####")
    print("Visual Dependent Figures: " + str(fig_vd["total"]))
    print("Total Figures: " + str(fig_all["total"]))

    print("##### Leaderboard Stats #####")

    table = [["", "Acc per figure (fAcc)", "Acc per question (aAcc)"],
              ["GPT Eval", figure_acc_gpt, q_acc_gpt]]
    leaderboard = PrettyTable(table[0])
    leaderboard.add_rows(table[1:])
    print(leaderboard)

