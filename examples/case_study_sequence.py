import yaml
import argparse
import torch
import torch.nn as nn
import os
import json
from copy import deepcopy
from tqdm.auto import tqdm

from models.tfm_model import TickTransformerModel
from models.tfm_model_rope import TickTransformerModelRope
from demoparser_utils.tick_tokenizer import TickTokenizer
from data.create_training_data import process_json_bytes, group_by_round


# =========================================================
# utils
# =========================================================

def load_config(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


def find_yaml(folder):
    for f in os.listdir(folder):
        if f.endswith(".yaml") and f != "tokenizer.yaml":
            return os.path.join(folder, f)
    raise RuntimeError(f"No yaml found in {folder}")


def load_checkpoint(folder):
    for f in os.listdir(folder):
        if f.endswith(".pth"):
            return os.path.join(folder, f)
    raise RuntimeError(f"No checkpoint found in {folder}")


# =========================================================
# heads
# =========================================================

class AliveHead(nn.Module):
    def __init__(self, dim, hidden, layers):
        super().__init__()
        net = []
        for _ in range(layers):
            net += [nn.Linear(dim, hidden), nn.GELU()]
            dim = hidden
        net.append(nn.Linear(dim, 10))
        self.head = nn.Sequential(*net)

    def forward(self, x):
        return self.head(x)


class KillHead(nn.Module):
    def __init__(self, dim, hidden, layers):
        super().__init__()
        net = []
        for _ in range(layers):
            net += [nn.Linear(dim, hidden), nn.GELU()]
            dim = hidden
        net.append(nn.Linear(dim, 22))
        self.head = nn.Sequential(*net)

    def forward(self, x):
        return self.head(x)


class WinRateHead(nn.Module):
    def __init__(self, dim, hidden, layers):
        super().__init__()
        net = []
        for _ in range(layers):
            net += [nn.Linear(dim, hidden), nn.GELU()]
            dim = hidden
        net.append(nn.Linear(dim, 1))
        self.head = nn.Sequential(*net)

    def forward(self, x):
        return self.head(x).squeeze(-1)


class DuelPredictionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden_layers, embedding_dim):
        super().__init__()

        layers = []
        prev_dim = input_dim + embedding_dim * 2

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.GELU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.head = nn.Sequential(*layers)
        self.embedding = nn.Embedding(10, embedding_dim)

    def forward(self, x, i, j):
        B = x.size(0)

        i = torch.full((B,), i, dtype=torch.long, device=x.device)
        j = torch.full((B,), j, dtype=torch.long, device=x.device)

        emb_i = self.embedding(i)
        emb_j = self.embedding(j)

        x = torch.cat([x, emb_i, emb_j], dim=-1)
        return self.head(x).squeeze(-1)


# =========================================================
# wrappers
# =========================================================

class PredictionModel(nn.Module):
    def __init__(self, base_model, head):
        super().__init__()
        self.base_model = base_model
        self.prediction_head = head

    def forward(self, x):
        feat = self.base_model.get_intermediate_data(x)
        last = feat[:, -1, :]
        mean = feat.mean(dim=1)
        tick_feat = torch.cat([last, mean], dim=-1)
        return self.prediction_head(tick_feat)


class DuelPredictionModel(nn.Module):
    def __init__(self, base_model, head):
        super().__init__()
        self.base_model = base_model
        self.prediction_head = head

    def forward(self, x, i, j):
        feat = self.base_model.get_intermediate_data(x)
        last = feat[:, -1, :]
        mean = feat.mean(dim=1)
        tick_feat = torch.cat([last, mean], dim=-1)
        return self.prediction_head(tick_feat, i, j)


# =========================================================
# load model
# =========================================================

def load_model(folder, head_type, device):
    yaml_path = find_yaml(folder)
    cfg = load_config(yaml_path)

    if cfg["model"]["model_name"] == "TickTransformerModel":
        base = TickTransformerModel(cfg["model"]).to(device)
    else:
        base = TickTransformerModelRope(cfg["model"]).to(device)

    embed = cfg["model"]["embed_dim"]

    if head_type == "alive":
        head = AliveHead(
            embed * 2,
            cfg["model"]["alive_hidden_dim"],
            cfg["model"]["alive_hidden_layers"]
        )
        model = PredictionModel(base, head)

    elif head_type == "kill":
        head = KillHead(
            embed * 2,
            cfg["model"]["nxt_kill_hidden_dim"],
            cfg["model"]["nxt_kill_hidden_layers"]
        )
        model = PredictionModel(base, head)

    elif head_type == "win":
        head = WinRateHead(
            embed * 2,
            cfg["model"]["win_rate_hidden_dim"],
            cfg["model"]["win_rate_hidden_layers"]
        )
        model = PredictionModel(base, head)

    elif head_type == "duel":
        head = DuelPredictionHead(
            embed * 2,
            cfg["model"]["duel_hidden_dim"],
            cfg["model"]["duel_hidden_layers"],
            cfg["model"]["duel_player_embedding_dim"]
        )
        model = DuelPredictionModel(base, head)

    else:
        raise ValueError(f"Unsupported head_type: {head_type}")

    ckpt = load_checkpoint(folder)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    return model.to(device), cfg


# =========================================================
# build padded input exactly like inference
# =========================================================

def build_input_window(round_tensor, tick_idx, ticks_per_sample, seq_len, pad_token):
    pad_front = max(0, ticks_per_sample - 1 - tick_idx)

    if pad_front > 0:
        pad = torch.full(
            (pad_front, round_tensor.shape[1]),
            pad_token,
            dtype=round_tensor.dtype
        )
        inp = torch.cat([pad, round_tensor[:tick_idx + 1]], dim=0)
    else:
        inp = round_tensor[tick_idx + 1 - ticks_per_sample:tick_idx + 1]

    if inp.shape[1] < seq_len:
        pad_len = seq_len - inp.shape[1]
        pad = torch.full(
            (inp.shape[0], pad_len),
            pad_token,
            dtype=inp.dtype
        )
        inp = torch.cat([inp, pad], dim=1)

    return inp[:, :seq_len]


# =========================================================
# prediction helpers
# =========================================================

def to_builtin(obj):
    """
    Convert tensors / numpy-like values into Python builtin types for json dump.
    """
    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_builtin(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


@torch.no_grad()
def predict_one_tick(
    inp,
    players_info,
    alive_model,
    kill_model,
    win_model,
    duel_model
):
    alive = torch.sigmoid(alive_model(inp))[0]          # [10]
    kill_logits = kill_model(inp)[0]                    # [22]
    win_rate = torch.sigmoid(win_model(inp))[0]         # scalar

    kill_prob = torch.softmax(kill_logits[:11], dim=0)  # [11]
    death_prob = torch.softmax(kill_logits[11:], dim=0) # [11]

    duel = torch.zeros(10, 10, device=inp.device)
    for i in range(10):
        for j in range(10):
            if i == j:
                duel[i, j] = 0.5
            else:
                duel[i, j] = torch.sigmoid(duel_model(inp, i, j))[0]

    # 组织成更易保存的结构
    alive_probs = alive.detach().cpu().tolist()
    kill_probs = kill_prob.detach().cpu().tolist()
    death_probs = death_prob.detach().cpu().tolist()
    duel_matrix = duel.detach().cpu().tolist()

    player_predictions = []
    for i, p in enumerate(players_info):
        player_predictions.append({
            "player_index": i,
            "name": p.get("name", f"player_{i}"),
            "team_num": p.get("team_num"),
            "is_alive_gt": p.get("is_alive"),
            "alive_prob": alive_probs[i],
            "next_killer_prob": kill_probs[i] if i < len(kill_probs) - 1 else None,
            "next_death_prob": death_probs[i] if i < len(death_probs) - 1 else None,
        })

    result = {
        "ct_win_rate": float(win_rate.item()),
        "alive_probs": alive_probs,
        "next_killer_probs": {
            "players": kill_probs[:10],
            "no_kill": kill_probs[10] if len(kill_probs) > 10 else None,
        },
        "next_death_probs": {
            "players": death_probs[:10],
            "no_death": death_probs[10] if len(death_probs) > 10 else None,
        },
        "duel_matrix": duel_matrix,
        "player_predictions": player_predictions,
    }

    return result


def attach_prediction_to_tick(tick_data, pred):
    """
    将预测结果挂到原始 tick 上，形成新的可保存结构。
    """
    out_tick = deepcopy(tick_data)
    out_tick["model_prediction"] = pred
    return out_tick


# =========================================================
# main
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--json_path", required=True)

    parser.add_argument("--alive_ckpt_dir", required=True)
    parser.add_argument("--kill_ckpt_dir", required=True)
    parser.add_argument("--winrate_ckpt_dir", required=True)
    parser.add_argument("--duel_ckpt_dir", required=True)

    parser.add_argument("--round_id", type=int, required=True)
    parser.add_argument("--start_sec", type=float, required=True)
    parser.add_argument("--end_sec", type=float, required=True)

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save predicted json"
    )

    parser.add_argument(
        "--save_mode",
        type=str,
        default="attach",
        choices=["attach", "prediction_only"],
        help="attach: save original tick + prediction; prediction_only: save only compact predictions"
    )

    parser.add_argument(
        "--tick_stride",
        type=int,
        default=1,
        help="Use every n-th tick in the selected range"
    )

    parser.add_argument("--device", default="cuda")

    parser.add_argument(
        "--remove_projectiles",
        action="store_true",
        help="Remove projectiles and grenade entities from json"
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    if args.end_sec < args.start_sec:
        raise ValueError("end_sec must be >= start_sec")

    print("Loading models...")
    alive_model, cfg = load_model(args.alive_ckpt_dir, "alive", device)
    kill_model, _ = load_model(args.kill_ckpt_dir, "kill", device)
    win_model, _ = load_model(args.winrate_ckpt_dir, "win", device)
    duel_model, _ = load_model(args.duel_ckpt_dir, "duel", device)
    print("Models loaded successfully.")

    print("Loading tokenizer...")
    tokenizer_path = os.path.join(args.alive_ckpt_dir, "tokenizer.yaml")
    with open(tokenizer_path, "r", encoding="utf-8-sig") as f:
        tokenizer_cfg = yaml.safe_load(f)
    tokenizer = TickTokenizer(tokenizer_cfg)
    valid_maps = set(tokenizer_cfg["maps"].keys())
    print("Tokenizer loaded successfully.")

    print("Loading json...")
    with open(args.json_path, "r", encoding="utf-8-sig") as f:
        json_data = json.load(f)

    if args.remove_projectiles:
        print("Removing projectiles / grenade entities from json...")
        for tick in json_data:
            tick["projectiles"] = []
            tick["entity_grenades"] = []

    print("Tokenizing json into round tensors...")
    round_tensors, _, _, _, _, _ = process_json_bytes(
        json.dumps(json_data).encode(),
        tokenizer,
        valid_maps
    )

    rounds = group_by_round(json_data)

    if args.round_id >= len(rounds):
        raise ValueError(
            f"Invalid round_id={args.round_id}, total rounds={len(rounds)}"
        )

    round_ticks = rounds[args.round_id]
    round_tensor = round_tensors[args.round_id]

    # 选出目标时间范围内的 tick
    # import pdb; pdb.set_trace()
    selected = []
    for idx, tick in enumerate(round_ticks):
        sec = tick["round_seconds"]
        if args.start_sec <= sec <= args.end_sec:
            selected.append((idx, tick))

    if len(selected) == 0:
        raise ValueError(
            f"No ticks found in round {args.round_id} between "
            f"{args.start_sec} and {args.end_sec} seconds."
        )

    selected = selected[::args.tick_stride]

    print(
        f"Selected {len(selected)} ticks in round {args.round_id} "
        f"from {args.start_sec:.3f}s to {args.end_sec:.3f}s "
        f"(stride={args.tick_stride})."
    )

    outputs = []
    total = len(selected)

    for n, (tick_idx, tick_data) in enumerate(
        tqdm(selected, desc="Predicting ticks", unit="tick"),
        start=1
    ):
        inp = build_input_window(
            round_tensor,
            tick_idx,
            cfg["data"]["ticks_per_sample"],
            cfg["data"]["seq_len"],
            cfg["data"]["pad_token"]
        ).unsqueeze(0).to(device)

        pred = predict_one_tick(
            inp=inp,
            players_info=tick_data["players_info"],
            alive_model=alive_model,
            kill_model=kill_model,
            win_model=win_model,
            duel_model=duel_model
        )

        if args.save_mode == "attach":
            out_item = attach_prediction_to_tick(tick_data, pred)
        else:
            out_item = {
                "round": tick_data.get("round"),
                "round_seconds": tick_data.get("round_seconds"),
                "tick_prediction": pred
            }

        outputs.append(to_builtin(out_item))

        if n % 20 == 0 or n == total:
            print(f"Processed {n}/{total} ticks...")

    final_output = {
        "meta": {
            "json_path": args.json_path,
            "round_id": args.round_id,
            "start_sec": args.start_sec,
            "end_sec": args.end_sec,
            "tick_stride": args.tick_stride,
            "save_mode": args.save_mode,
            "remove_projectiles": args.remove_projectiles,
            "num_ticks": len(outputs),
        },
        "results": outputs
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True) if os.path.dirname(args.output_path) else None

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print(f"Saved predictions to: {args.output_path}")


if __name__ == "__main__":
    main()