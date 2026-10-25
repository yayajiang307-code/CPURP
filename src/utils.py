from collections import defaultdict, deque
import multiprocessing as mp

import numpy as np
import scipy.sparse as sp


MAX_MP_WORKERS = 8
MAX_PATHS_PER_PAIR = 50000
MAX_BFS_EXPANSIONS = 200000
MAX_BFS_QUEUE_SIZE = 50000


def count_all_paths_with_mp(e2re, max_path_len, head2tails):
    if not head2tails:
        return defaultdict(set)

    n_cores, pool, range_list = get_params_for_mp(len(head2tails))
    try:
        results = pool.map(
            count_all_paths,
            zip(
                [e2re] * n_cores,
                [max_path_len] * n_cores,
                [head2tails[i[0]:i[1]] for i in range_list],
                range(n_cores),
            ),
        )
    finally:
        pool.close()
        pool.join()

    res = defaultdict(set)
    for ht2paths in results:
        res.update(ht2paths)

    return res


def get_params_for_mp(n_triples):
    n_cores = max(1, min(mp.cpu_count(), n_triples, MAX_MP_WORKERS))
    pool = mp.Pool(n_cores)
    avg = n_triples // n_cores

    range_list = []
    start = 0
    for i in range(n_cores):
        num = avg + 1 if i < n_triples - avg * n_cores else avg
        range_list.append([start, start + num])
        start += num

    return n_cores, pool, range_list


# input: [(h1, {t1, t2 ...}), (h2, {t3 ...}), ...]
# output: {(h1, t1): paths, (h1, t2): paths, (h2, t3): paths, ...}
def count_all_paths(inputs):
    e2re, max_path_len, head2tails, pid = inputs
    ht2paths = {}
    for i, (head, tails) in enumerate(head2tails):
        ht2paths.update(bfs(head, tails, e2re, max_path_len))
        print('pid %d:  %d / %d' % (pid, i, len(head2tails)))
    print('pid %d  done' % pid)
    return ht2paths


def bfs(head, tails, e2re, max_path_len):
    # Bounded BFS that keeps only the queue frontier and discovered paths.
    # This avoids the unbounded growth of `all_paths` in dense graphs.
    if head not in e2re:
        return defaultdict(set)

    queue = deque([(head, [], (head,))])
    expansions = 0
    paths_found = 0
    ht2paths = defaultdict(set)

    while queue:
        if len(queue) > MAX_BFS_QUEUE_SIZE or expansions > MAX_BFS_EXPANSIONS or paths_found >= MAX_PATHS_PER_PAIR:
            break

        node, rel_seq, visited_nodes = queue.popleft()
        if len(rel_seq) >= max_path_len:
            continue

        for edge in e2re[node]:
            expansions += 1
            if expansions > MAX_BFS_EXPANSIONS or paths_found >= MAX_PATHS_PER_PAIR:
                break
            if edge[1] in visited_nodes:
                continue

            new_rel_seq = rel_seq + [edge[0]]
            new_visited_nodes = visited_nodes + (edge[1],)

            if edge[1] in tails:
                ht2paths[(head, edge[1])].add(tuple(new_rel_seq))
                paths_found += 1
            elif len(new_rel_seq) < max_path_len:
                queue.append((edge[1], new_rel_seq, new_visited_nodes))

    return ht2paths


def count_paths(triplets, ht2paths, train_set):
    res = []

    for head, tail, relation in triplets:
        path_set = ht2paths[(head, tail)]
        if (tail, head, relation) in train_set:
            path_list = list(path_set)
        else:
            path_list = list(path_set - {tuple([relation])})
        res.append([list(i) for i in path_list])

    return res


def get_path_dict_and_length(train_paths, valid_paths, test_paths, null_relation, max_path_len):
    path2id = {}
    id2path = []
    id2length = []
    n_paths = 0

    for paths_of_triplet in train_paths + valid_paths + test_paths:
        for path in paths_of_triplet:
            path_tuple = tuple(path)
            if path_tuple not in path2id:
                path2id[path_tuple] = n_paths
                id2length.append(len(path))
                id2path.append(path + [null_relation] * (max_path_len - len(path)))  # padding
                n_paths += 1
    return path2id, id2path, id2length


def one_hot_path_id(train_paths, valid_paths, test_paths, path_dict):
    res = []
    for data in (train_paths, valid_paths, test_paths):
        bop_list = []  # bag of paths
        for paths in data:
            bop_list.append([path_dict[tuple(path)] for path in paths])
        res.append(bop_list)

    return [get_sparse_feature_matrix(bop_list, len(path_dict)) for bop_list in res]


def sample_paths(train_paths, valid_paths, test_paths, path_dict, path_samples):
    res = []
    for data in [train_paths, valid_paths, test_paths]:
        path_ids_for_data = []
        for paths in data:
            path_ids_for_triplet = [path_dict[tuple(path)] for path in paths]
            sampled_path_ids_for_triplet = np.random.choice(
                path_ids_for_triplet, size=path_samples, replace=len(path_ids_for_triplet) < path_samples)
            path_ids_for_data.append(sampled_path_ids_for_triplet)

        path_ids_for_data = np.array(path_ids_for_data, dtype=np.int32)
        res.append(path_ids_for_data)
    return res


def get_sparse_feature_matrix(non_zeros, n_cols):
    features = sp.lil_matrix((len(non_zeros), n_cols), dtype=np.float64)
    for i in range(len(non_zeros)):
        for j in non_zeros[i]:
            features[i, j] = +1.0
    return features


def sparse_to_tuple(sparse_matrix):
    if not sp.isspmatrix_coo(sparse_matrix):
        sparse_matrix = sparse_matrix.tocoo()
    indices = np.vstack((sparse_matrix.row, sparse_matrix.col)).transpose()
    values = sparse_matrix.data
    shape = sparse_matrix.shape
    return indices, values, shape
