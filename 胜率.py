# 胜率.py
import random
from collections import Counter
from typing import List, Tuple, Dict, Optional
import argparse

# ---------------------------
# 牌堆 & 工具函数（与你给的代码兼容）
# ---------------------------
FULL_RANKS = ['2', '3', '4', '5', '6', '7', '8', '9',
              '10', 'J', 'Q', 'K', 'A']
SUITS = ['♠', '♥', '♣', '♦']
rank_value = {r: i for i, r in enumerate(FULL_RANKS, start=2)}

CATEGORY_RANK = {
    "单张": 1,
    "对子": 2,
    "顺子": 3,
    "同花": 4,
    "顺金": 5,
    "豹子": 6
}


def build_deck(ranks: Optional[List[str]] = None) -> List[str]:
    """
    构造牌堆
    ranks: None 表示整副牌（52张），或传入子集如 ['9','10','J','Q','K','A']
    """
    if ranks is None:
        ranks = FULL_RANKS
    return [r + s for r in ranks for s in SUITS]


def classify(hand: List[str]) -> str:
    """判断牌型（与你给的逻辑保持一致，包含 A-2-3 特殊顺子）"""
    rs = [card[:-1] for card in hand]  # 点数
    ss = [card[-1] for card in hand]  # 花色
    vs = sorted([rank_value[r] for r in rs])

    cnt = Counter(rs)
    is_flush = len(set(ss)) == 1

    # 普通顺子：最大-最小==2 且三张互不相同
    is_straight = (vs[2] - vs[0] == 2 and len(set(vs)) == 3)

    # 特殊顺子 A,2,3
    if set(rs) == {'A', '2', '3'}:
        is_straight = True

    if 3 in cnt.values():
        return "豹子"
    if is_flush and is_straight:
        return "顺金"
    if is_flush:
        return "同花"
    if is_straight:
        return "顺子"
    if 2 in cnt.values():
        return "对子"
    return "单张"


def _straight_top(values: List[int]) -> Optional[int]:
    """
    若是顺子，返回顺子的最高牌点数（A-2-3 视为 3）；否则返回 None
    """
    s = set(values)
    if s == {rank_value['A'], rank_value['2'], rank_value['3']}:
        return rank_value['3']  # A-2-3 作为最小顺子，顶牌按 3
    if len(s) == 3:
        vs = sorted(s)
        if vs[2] - vs[0] == 2:
            return vs[2]
    return None


def hand_key(hand: List[str]) -> Tuple[int, Tuple[int, int, int]]:
    """
    生成可比较的手牌强度键：(牌型等级, 细分比较键)
    细分比较键规则（仅比较点数，不比较花色）：
      - 豹子:     (三条点数, 0, 0)
      - 顺金/顺子: (顺子顶牌, 0, 0)；A-2-3 顶牌为 3
      - 同花/单张: 按点数降序排列 (a, b, c)
      - 对子:     (对子点数, 踢脚点数, 0)
    """
    cat = classify(hand)
    rs = [card[:-1] for card in hand]
    vs = sorted([rank_value[r] for r in rs], reverse=True)
    cnt = Counter(vs)

    if cat == "豹子":
        triple = max(v for v, c in cnt.items() if c == 3)
        key = (triple, 0, 0)

    elif cat in ("顺金", "顺子"):
        top = _straight_top(vs[::-1])  # 传入无所谓顺序，用集合判断
        # 保险起见再处理 A-2-3
        if set(rs) == {'A', '2', '3'}:
            top = rank_value['3']
        assert top is not None
        key = (top, 0, 0)

    elif cat == "对子":
        pair = max(v for v, c in cnt.items() if c == 2)
        kicker = max(v for v, c in cnt.items() if c == 1)
        key = (pair, kicker, 0)

    else:  # 同花 或 单张
        vs_sorted = tuple(sorted(vs, reverse=True))
        key = (vs_sorted[0], vs_sorted[1], vs_sorted[2])

    return (CATEGORY_RANK[cat], key)


# ---------------------------
# 条件胜率统计（关键：按“拿到该牌型时是否赢”来算）
# ---------------------------
def simulate_conditional_winrate(players: int = 3,
                                 rounds: int = 100000,
                                 ranks: Optional[List[str]] = None,
                                 seed: Optional[int] = None,
                                 tie_as_win: bool = True) -> Dict[str, Dict[str, float]]:
    """
    统计“条件胜率”：胜率 = 某牌型赢的次数 / 某牌型出现的次数
    - 每局对每位玩家单独统计
    - tie_as_win=True 时，平局（并列最大）计为赢；否则不计入“赢”
    返回字典：
      {
        牌型: {
          "appear": 出现次数,
          "appear_freq": 出现频率(占所有手数),
          "win": 获胜次数,
          "win_rate": 胜率(赢/出现) 或 0/NaN
        }, ...
      }
    """
    if seed is not None:
        random.seed(seed)

    deck = build_deck(ranks)
    appear = Counter()
    wins = Counter()

    for _ in range(rounds):
        # 发牌
        cards = random.sample(deck, players * 3)
        hands = [cards[i * 3:(i + 1) * 3] for i in range(players)]

        # 本局每手的强度
        keys = [hand_key(h) for h in hands]
        max_key = max(keys)
        # 找到本局的赢家（可能多人并列）
        winners = [i for i, k in enumerate(keys) if k == max_key]

        # 逐手统计出现 &（可选）胜利
        for i, h in enumerate(hands):
            cat = classify(h)
            appear[cat] += 1
            if tie_as_win:
                if i in winners:
                    wins[cat] += 1
            else:
                if len(winners) == 1 and i == winners[0]:
                    wins[cat] += 1

    total_hands = rounds * players
    result: Dict[str, Dict[str, float]] = {}
    for cat in CATEGORY_RANK.keys():
        a = appear[cat]
        w = wins[cat]
        appear_freq = a / total_hands if total_hands > 0 else 0.0
        win_rate = (w / a) if a > 0 else float('nan')  # a==0 时为 NaN
        result[cat] = {
            "appear": float(a),
            "appear_freq": appear_freq,
            "win": float(w),
            "win_rate": win_rate
        }
    return result


# ---------------------------
# 命令行入口
# ---------------------------
def parse_ranks_arg(ranks_arg: str) -> Optional[List[str]]:
    """
    ranks 参数解析：
      - 'full' 使用整副牌
      - '9A'   使用 9,10,J,Q,K,A
      - 逗号分隔自定义，如 '8,9,10,J,Q,K,A'
    """
    if ranks_arg.lower() == 'full':
        return None
    if ranks_arg.lower() == '9a':
        return ['9', '10', 'J', 'Q', 'K', 'A']
    parts = [p.strip() for p in ranks_arg.split(',') if p.strip()]
    if parts:
        # 合法性粗检
        for p in parts:
            if p not in FULL_RANKS:
                raise ValueError(f"非法点数: {p}")
        return parts
    return None


def main():
    parser = argparse.ArgumentParser(description="n 人炸金花各牌型条件胜率模拟")
    parser.add_argument('--players', type=int, default=6, help='玩家人数（>=2）')
    parser.add_argument('--rounds', type=int, default=100000, help='模拟局数')
    parser.add_argument('--ranks', type=str, default='9,10,J,Q,K,A',
                        help="牌值集合: 'full' | '9A' | '逗号分隔自定义(如 9,10,J,Q,K,A)'")
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--strict_tie', action='store_true',
                        help='平局不计为赢（默认平局算赢）')
    args = parser.parse_args()

    ranks = parse_ranks_arg(args.ranks)
    res = simulate_conditional_winrate(players=args.players,
                                       rounds=args.rounds,
                                       ranks=ranks,
                                       seed=args.seed,
                                       tie_as_win=not args.strict_tie)

    # 输出（按牌型强度从高到低）
    print(f"\n玩家数: {args.players}，局数: {args.rounds}，"
          f"牌集合: {('整副牌' if ranks is None else ','.join(ranks))}，"
          f"{'平局算赢' if not args.strict_tie else '平局不算赢'}")
    header = f"{'牌型':<4}  {'出现次数':>10}  {'出现频率':>10}  {'获胜次数':>10}  {'胜率':>10}"
    print(header)
    print("-" * len(header))
    for cat in sorted(CATEGORY_RANK.keys(), key=lambda c: CATEGORY_RANK[c], reverse=True):
        a = res[cat]["appear"]
        af = res[cat]["appear_freq"]
        w = res[cat]["win"]
        wr = res[cat]["win_rate"]
        wr_str = ("N/A" if wr != wr else f"{wr:.4%}")  # NaN 检测
        print(f"{cat:<4}  {int(a):>10}  {af:>9.4%}  {int(w):>10}  {wr_str:>10}")


if __name__ == "__main__":
    main()
# 检测新分支