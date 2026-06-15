from mlx_streaming.tools.crosstoken_recall import crosstoken_recall


def test_crosstoken_recall_n1_scores_from_previous_occurrence():
    # 同一层 L=0 的三次出现（执行顺序）。首次无历史→不计分。
    events = [
        {"layer": 0, "experts": [1, 2, 3], "miss": [1, 2, 3]},
        {"layer": 0, "experts": [2, 3, 4], "miss": [4]},   # pred(前1次)={1,2,3}
        {"layer": 0, "experts": [3, 4, 5], "miss": [5]},   # pred(前1次)={2,3,4}
    ]
    r = crosstoken_recall(events, history_n=1)
    assert r["n_scored"] == 2
    # full = (|{1,2,3}∩{2,3,4}|=2 + |{2,3,4}∩{3,4,5}|=2)/(3+3) = 4/6
    assert r["recall_full"] == 0.6667
    # miss = ({4}覆盖0 + {5}覆盖0)/2 = 0 —— 历史覆盖不到新颖 miss
    assert r["recall_miss"] == 0.0
    assert r["tot_miss"] == 2


def test_crosstoken_recall_n2_unions_two_previous_occurrences():
    events = [
        {"layer": 0, "experts": [1], "miss": []},
        {"layer": 0, "experts": [2], "miss": []},
        {"layer": 0, "experts": [1, 2], "miss": [1, 2]},  # pred=union({1},{2})={1,2}
    ]
    r = crosstoken_recall(events, history_n=2)
    assert r["n_scored"] == 2
    # 复发 miss 被前两次并集覆盖：{1,2}∩{1,2} = 2/2
    assert r["recall_miss"] == 1.0
