# .demo -> .json
from demoparser_utils.state_extract import extract_states, save_as_json
from demoparser2 import DemoParser
import numpy as np
import argparse
from pathlib import Path
import time
import json
import gzip


def get_total_rounds(parser: DemoParser):
    """
    获取整场 demo 的总回合数（按 total_rounds_played 推断）。
    """
    df = parser.parse_ticks(wanted_props=["total_rounds_played"])
    if df is None or len(df) == 0:
        return 0
    return int(df["total_rounds_played"].max()) + 1


def get_important_ticks(parser: DemoParser, interval=0.5, round_id=None):
    """
    Sample ticks every `interval` seconds INSIDE EACH ROUND.

    Args:
        parser: DemoParser
        interval: sampling interval in seconds
        round_id: if not None, only sample this round

    Returns:
        list[int]
    """

    df = parser.parse_ticks(
        wanted_props=[
            "game_time",
            "total_rounds_played",
        ]
    )

    # deduplicate (10 players per tick)
    df = df[["tick", "game_time", "total_rounds_played"]] \
            .drop_duplicates() \
            .sort_values("tick")

    all_ticks = []

    round_starts = parser.parse_event("round_freeze_end")
    round_start_ticks = round_starts["tick"].to_numpy().astype(int).tolist()
    round_start_ticks.sort()

    df_round_starts_time = [None for _ in range(df["total_rounds_played"].max() + 1)]

    round_start_round_ids = df[df["tick"].isin(round_start_ticks)]["total_rounds_played"].to_numpy()
    round_start_game_times = df[df["tick"].isin(round_start_ticks)]["game_time"].to_numpy()

    for idx, rid in enumerate(round_start_round_ids):
        df_round_starts_time[int(rid)] = round_start_game_times[idx]

    df_round_starts_time = np.array(df_round_starts_time, dtype=object)
    df["round_start_time"] = df_round_starts_time[df["total_rounds_played"].to_numpy()]

    # process per round
    for rid, df_round in df.groupby("total_rounds_played"):
        rid = int(rid)

        # 如果指定了 round_id，则只处理该回合
        if round_id is not None and rid != round_id:
            continue

        times = df_round["game_time"].to_numpy()
        round_start = df_round["round_start_time"].iloc[-1]
        ticks = df_round["tick"].to_numpy()

        if round_start is None:
            continue

        round_seconds = times - round_start

        if len(round_seconds) == 0:
            continue

        t_end = round_seconds[-1]
        target_times = np.arange(0.5, t_end, interval)

        idx = 0
        for t in target_times:
            while idx + 1 < len(round_seconds) and round_seconds[idx + 1] < t:
                idx += 1

            if idx + 1 < len(round_seconds):
                if abs(round_seconds[idx + 1] - t) < abs(round_seconds[idx] - t):
                    all_ticks.append(int(ticks[idx + 1]))
                else:
                    all_ticks.append(int(ticks[idx]))
            else:
                all_ticks.append(int(ticks[idx]))

    return sorted(set(all_ticks))


def save_round_aligned_json(states, output_path, total_rounds, round_id, compression=False):
    """
    将 states 组织为“按回合对齐”的 JSON：
    未解析的回合置空列表，解析的目标回合填入实际 states。

    输出示例：
    [
      [],
      [],
      [... round 2 states ...],
      [],
      ...
    ]

    注意：
    - 这里不改变单个 state 的内容格式，只改变最外层组织方式。
    - 如果你原本的下游逻辑依赖 flat list，请不要开启这个模式。
    """
    round_aligned = [[] for _ in range(total_rounds)]

    if round_id is not None:
        if round_id < 0 or round_id >= total_rounds:
            raise ValueError(f"round_id={round_id} out of range, total_rounds={total_rounds}")
        round_aligned[round_id] = states
    else:
        # 如果没有指定 round_id，则尽量按 state 里的 round 字段分组
        # 如果 state 中没有 round 字段，则全部放到第 0 回合（保底）
        for s in states:
            if isinstance(s, dict) and "round" in s:
                rid = int(s["round"])
                if 0 <= rid < total_rounds:
                    round_aligned[rid].append(s)
            else:
                if total_rounds > 0:
                    round_aligned[0].append(s)

    if compression:
        with gzip.open(output_path, "wt", encoding="utf-8") as f:
            json.dump(round_aligned, f, ensure_ascii=False)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(round_aligned, f, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Process a CS demo into JSON states")
    parser.add_argument("-path", type=str, required=True, help="Path to .demo file")
    parser.add_argument("-interval", type=float, default=0.5, help="Sampling interval in seconds")
    parser.add_argument("-out", type=str, required=True, help="Output JSON file path")
    parser.add_argument("-debug", type=bool, required=False, default=0, help="Output processing information")
    parser.add_argument("-compression", type=bool, required=False, default=0, help="Use compression for JSON output")

    # 新增：只解析单个回合
    parser.add_argument(
        "-round_id",
        type=int,
        required=False,
        default=None,
        help="Only parse the specified round id. If omitted, parse all rounds."
    )

    # 新增：是否将未解析回合置空并按回合索引对齐保存
    parser.add_argument(
        "-pad_empty_rounds",
        type=bool,
        required=False,
        default=0,
        help="When round_id is set, save as round-aligned JSON and leave other rounds empty."
    )

    args = parser.parse_args()

    if args.debug:
        print(args)

    start_time = time.perf_counter()

    demo_path = Path(args.path)
    output_path = Path(args.out)
    interval = args.interval
    debug = args.debug
    round_id = args.round_id
    pad_empty_rounds = args.pad_empty_rounds

    if not demo_path.exists():
        print(f"Error: {demo_path} does not exist")
        return

    if debug:
        print(f"Parsing demo: {demo_path}")

    demo_parser = DemoParser(str(demo_path))
    total_rounds = get_total_rounds(demo_parser)

    if round_id is not None:
        if round_id < 0 or round_id >= total_rounds:
            print(f"Error: round_id={round_id} out of range, total_rounds={total_rounds}")
            return

    if debug:
        if round_id is None:
            print(f"Sampling ticks every {interval} seconds per round...")
        else:
            print(f"Sampling ticks every {interval} seconds for round {round_id} only...")

    ticks = get_important_ticks(demo_parser, interval=interval, round_id=round_id)

    if debug:
        print(f"Extracting states for {len(ticks)} ticks...")

    states = extract_states(str(demo_path), ticks)

    if debug:
        print(f"Saving states to {output_path}..., compression={args.compression}")

    # 情况1：不指定单回合，保持原逻辑
    if round_id is None:
        save_as_json(states, str(output_path), compression=args.compression)

    # 情况2：指定单回合，但不补空，仍然保持“原始保存格式”
    elif not pad_empty_rounds:
        save_as_json(states, str(output_path), compression=args.compression)

    # 情况3：指定单回合，并按回合对齐，其他回合置空
    else:
        save_round_aligned_json(
            states=states,
            output_path=str(output_path),
            total_rounds=total_rounds,
            round_id=round_id,
            compression=args.compression
        )

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    if debug:
        print(f"Total rounds in demo: {total_rounds}")
        if output_path.exists():
            print(f"Done. The size of output file is {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"Total processing time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()