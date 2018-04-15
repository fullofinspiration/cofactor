# coding: utf-8
if __name__ == '__main__':
    # # Fit CoFactor model to the binarized ML20M
    import itertools
    import glob
    import os
    import sys

    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    import numpy as np
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy import sparse
    import seaborn as sns

    sns.set(context="paper", font_scale=1.5, rc={"lines.linewidth": 2}, font='DejaVu Serif')
    import sys

    sys.path.append('E:\workspace\source_code\cofactor\src')
    import cofacto
    import rec_eval

    # ### Construct the positive pairwise mutual information (PPMI) matrix
    # Change this to wherever you saved the pre-processed data following [this notebook](./preprocess_ML20M.ipynb).
    # 修改处
    # DATA_DIR = 'E:\datasets\ml-1m\pro'
    DATA_DIR = 'E:\datasets\ml-1m\pro'
    unique_uid = list()
    with open(os.path.join(DATA_DIR, 'unique_uid.txt'), 'r') as f:
        for line in f:
            unique_uid.append(line.strip())
    unique_sid = list()
    with open(os.path.join(DATA_DIR, 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())
    n_items = len(unique_sid)
    n_users = len(unique_uid)
    print n_users, n_items


    def load_data(csv_file, shape=(n_users, n_items)):
        tp = pd.read_csv(csv_file)
        timestamps, rows, cols = np.array(tp['timestamp']), np.array(tp['uid']), np.array(tp['sid'])
        seq = np.concatenate((rows[:, None], cols[:, None], np.ones((rows.size, 1), dtype='int'), timestamps[:, None]),
                             axis=1)
        data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int16, shape=shape)
        return data, seq


    train_data, train_raw = load_data(os.path.join(DATA_DIR, 'train.csv'))
    watches_per_movie = np.asarray(train_data.astype('int64').sum(axis=0)).ravel()
    print("The mean (median) watches per movie is %d (%d)" % (watches_per_movie.mean(), np.median(watches_per_movie)))
    user_activity = np.asarray(train_data.sum(axis=1)).ravel()
    print("The mean (median) movies each user wathced is %d (%d)" % (user_activity.mean(), np.median(user_activity)))
    vad_data, vad_raw = load_data(os.path.join(DATA_DIR, 'validation.csv'))

    '''
    plt.semilogx(1 + np.arange(n_users), -np.sort(-user_activity), 'o')
    plt.ylabel('Number of items that this user clicked on')
    plt.xlabel('User rank by number of consumed items')
    pass

    plt.semilogx(1 + np.arange(n_items), -np.sort(-watches_per_movie), 'o')
    plt.ylabel('Number of users who watched this movie')
    plt.xlabel('Movie rank by number of watches')
    plt.show()
    pass
    '''

    # ### Generate co-occurrence matrix based on the user's entire watching history
    from joblib import Parallel, delayed
    import solve_co_user

    batch_size = 400
    start_idx = range(0, n_users, batch_size)
    end_idx = start_idx[1:] + [n_users]
    # train_data_T = train_data.T.tocsr()
    solve_co_user.use_coord_batch(start_idx, end_idx, train_data, DATA_DIR)
    '''
    for lo, hi in zip(start_idx, end_idx):  
            _coord_batch(lo, hi, train_data, DATA_DIR)

    pass
    '''
    X = sparse.csr_matrix((n_items, n_items), dtype='float32')
    for lo, hi in zip(start_idx, end_idx):
        coords = np.load(os.path.join(DATA_DIR, 'coo_%d_%d.npy' % (lo, hi)))
        rows = coords[:, 0]
        cols = coords[:, 1]
        tmp = sparse.coo_matrix((np.ones_like(rows), (rows, cols)), shape=(n_items, n_items), dtype='float32').tocsr()
        X = X + tmp
        print("User %d to %d finished" % (lo, hi))
        sys.stdout.flush()

    # Note: Don't forget to delete all the temporary coo_LO_HI.npy files
    np.save(os.path.join(DATA_DIR, 'coordinate_co_binary_data.npy'), X.data)
    np.save(os.path.join(DATA_DIR, 'coordinate_co_binary_indices.npy'), X.indices)
    np.save(os.path.join(DATA_DIR, 'coordinate_co_binary_indptr.npy'), X.indptr)
    '''
    float(X.nnz) / np.prod(X.shape)
    '''
    # ### Or load the pre-saved co-occurrence matrix
    # or co-occurrence matrix from the entire user history
    dir_predix = DATA_DIR
    data = np.load(os.path.join(dir_predix, 'coordinate_co_binary_data.npy'))
    indices = np.load(os.path.join(dir_predix, 'coordinate_co_binary_indices.npy'))
    indptr = np.load(os.path.join(dir_predix, 'coordinate_co_binary_indptr.npy'))
    X = sparse.csr_matrix((data, indices, indptr), shape=(n_items, n_items))
    float(X.nnz) / np.prod(X.shape)


    def get_row(Y, i):
        lo, hi = Y.indptr[i], Y.indptr[i + 1]
        return lo, hi, Y.data[lo:hi], Y.indices[lo:hi]


    count = np.asarray(X.sum(axis=1)).ravel()
    n_pairs = X.data.sum()

    # ### Construct the SPPMI matrix
    M = X.copy()
    for i in xrange(n_items):
        lo, hi, d, idx = get_row(M, i)
        M.data[lo:hi] = np.log(d * n_pairs / (count[i] * count[idx]))
    M.data[M.data < 0] = 0
    M.eliminate_zeros()
    print float(M.nnz) / np.prod(M.shape)

    # Now $M$ is the PPMI matrix. Depending on the number of negative examples $k$, we can obtain the shifted PPMI matrix as $\max(M_{wc} - \log k, 0)$
    # number of negative samples
    k_ns = 1
    M_ns = M.copy()
    if k_ns > 1:
        offset = np.log(k_ns)
    else:
        offset = 0.
    M_ns.data -= offset
    M_ns.data[M_ns.data < 0] = 0
    M_ns.eliminate_zeros()
    '''
    plt.hist(M_ns.data, bins=50)
    plt.yscale('log')
    pass

    '''
    float(M_ns.nnz) / np.prod(M_ns.shape)

    # ### Train the model
    scale = 0.03
    n_components = 10
    max_iter = 20
    n_jobs = 8
    lam_theta = lam_beta = 1e-5 * scale
    lam_gamma = 1e-5
    c0 = 1. * scale
    c1 = 10. * scale
    save_dir = os.path.join(DATA_DIR, 'ML20M_ns%d_scale%1.2E' % (k_ns, scale))
    reload(cofacto)
    coder = cofacto.CoFacto(n_components=n_components, max_iter=max_iter, batch_size=1000, init_std=0.01, n_jobs=n_jobs,
                            random_state=98765, save_params=True, save_dir=save_dir, early_stopping=True, verbose=True,
                            lam_theta=lam_theta, lam_beta=lam_beta, lam_gamma=lam_gamma, c0=c0, c1=c1)
    # import ipdb; ipdb.set_trace()  # <--- *BAMF!*
    coder.fit(train_data, M_ns, vad_data=vad_data, batch_users=5000, k=100)
    # import ipdb; ipdb.set_trace()  # <--- *BAMF!*
    test_data, _ = load_data(os.path.join(DATA_DIR, 'test.csv'))
    test_data.data = np.ones_like(test_data.data)
    # import ipdb; ipdb.set_trace()  # <--- *BAMF!*
    n_params = len(glob.glob(os.path.join(save_dir, '*.npz')))
    # import ipdb; ipdb.set_trace()  # <--- *BAMF!*
    params = np.load(os.path.join(save_dir, 'CoFacto_K%d_iter%d.npz' % (n_components, n_params - 1)))
    # import ipdb; ipdb.set_trace()  # <--- *BAMF!*
    U, V = params['U'], params['V']
    # import ipdb; ipdb.set_trace()  # <--- *BAMF!*
    # print 'Test Recall@20: %.4f' % rec_eval.recall_at_k(train_data, test_data, U, V, k=20, vad_data=vad_data)
    # import ipdb; ipdb.set_trace()  # <--- *BAMF!*
    # print 'Test Recall@50: %.4f' % rec_eval.recall_at_k(train_data, test_data, U, V, k=50, vad_data=vad_data)
    # import ipdb; ipdb.set_trace()  # <--- *BAMF!*+
    # print 'Test NDCG@100: %.4f' % rec_eval.normalized_dcg_at_k(train_data, test_data, U, V, k=100, vad_data=vad_data)
    # import ipdb; ipdb.set_trace()  # <--- *BAMF!*
    # print 'Test MAP@100: %.4f' % rec_eval.map_at_k(train_data, test_data, U, V, k=100, vad_data=vad_data)
    # np.savez('CoFactor_K100_ML20M.npz', U=U, V=V)
