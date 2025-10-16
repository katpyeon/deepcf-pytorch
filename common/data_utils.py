"""
데이터 로딩 및 변환 유틸리티

이 모듈은 DeepCF 형식의 데이터를 로드하고 다양한 데이터 형식 간 변환을 제공합니다.
이 유틸리티는 DMF, MLP, CFNet 모델에서 공유됩니다.

Functions:
    load_deepcf_data: DeepCF 형식의 데이터 로드
    load_cornac_data_with_full_space: Cornac용 데이터 로드 (전체 item 공간 유지)
    deepcf_to_uir: DeepCF 형식을 Cornac UIR 튜플로 변환
    get_train_matrix: 희소 행렬을 밀집 numpy 배열로 변환
"""

import numpy as np
import scipy.sparse as sp


def load_deepcf_data(data_path, dataset_name):
    """
    DeepCF 형식의 데이터를 로드합니다.

    Args:
        data_path (str): 데이터 디렉토리 경로
        dataset_name (str): 데이터셋 이름 (예: 'ml-1m', 'ml-1m-sample100')

    Returns:
        tuple: (train_matrix, testRatings, testNegatives, num_users, num_items)
            - train_matrix: scipy.sparse.dok_matrix
            - testRatings: [user, item] 쌍의 리스트
            - testNegatives: 부정 아이템 리스트의 리스트
            - num_users: int
            - num_items: int
    """
    train_file = f"{data_path}{dataset_name}.train.rating"
    test_file = f"{data_path}{dataset_name}.test.rating"
    neg_file = f"{data_path}{dataset_name}.test.negative"

    # Train matrix 로드
    num_users, num_items = 0, 0
    with open(train_file, 'r') as f:
        for line in f:
            if line.strip():
                arr = line.strip().split('\t')
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)

    train_matrix = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
    with open(train_file, 'r') as f:
        for line in f:
            if line.strip():
                arr = line.strip().split('\t')
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if rating > 0:
                    train_matrix[user, item] = 1.0

    # Test ratings 로드
    testRatings = []
    with open(test_file, 'r') as f:
        for line in f:
            if line.strip():
                arr = line.strip().split('\t')
                testRatings.append([int(arr[0]), int(arr[1])])

    # Test negatives 로드
    testNegatives = []
    with open(neg_file, 'r') as f:
        for line in f:
            if line.strip():
                arr = line.strip().split('\t')
                negatives = [int(x) for x in arr[1:]]
                testNegatives.append(negatives)

    return train_matrix, testRatings, testNegatives, num_users + 1, num_items + 1


def load_cornac_data_with_full_space(data_path, dataset_name):
    """
    Cornac 평가를 위해 전체 item 공간을 유지하며 데이터를 로드합니다.

    이 함수는 DeepCF의 load_deepcf_data()와 동일한 방식으로 전체 item 공간을 계산합니다.
    Train과 Test 파일을 모두 읽어 전체 user/item ID를 추적하여 num_items를 결정합니다.

    **중요**: Cornac의 RatioSplit은 train set의 item 수만 사용하지만,
    이 함수는 train+test의 전체 item 공간을 반영하여 pretrain 모델과 차원을 일치시킵니다.

    Args:
        data_path (str): 데이터 디렉토리 경로
        dataset_name (str): 데이터셋 이름 (예: 'ml-1m', 'ml-1m-sample100')

    Returns:
        tuple: (train_data, test_data, num_users, num_items)
            - train_data: [(user_id, item_id, rating), ...] for train
            - test_data: [(user_id, item_id, rating), ...] for test
            - num_users: 전체 유저 수
            - num_items: 전체 아이템 수 (train + test에 나타난 모든 item)

    Example:
        >>> train_data, test_data, num_users, num_items = load_cornac_data_with_full_space(
        ...     '../datasets/', 'ml-1m-sample100'
        ... )
        >>> print(f"Users: {num_users}, Items: {num_items}")
        Users: 100, Items: 2591
    """
    train_file = f"{data_path}{dataset_name}.train.rating"
    test_file = f"{data_path}{dataset_name}.test.rating"

    # 전체 user/item ID 추적
    all_users = set()
    all_items = set()

    # Train data 읽기
    train_data = []
    with open(train_file, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split('\t')
                user_id, item_id, rating = parts[0], parts[1], float(parts[2])
                train_data.append((user_id, item_id, rating))
                all_users.add(user_id)
                all_items.add(item_id)

    # Test data 읽기
    test_data = []
    with open(test_file, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split('\t')
                user_id, item_id, rating = parts[0], parts[1], float(parts[2])
                test_data.append((user_id, item_id, rating))
                all_users.add(user_id)
                all_items.add(item_id)  # ← 핵심: test에만 있는 item도 포함!

    num_users = len(all_users)
    num_items = len(all_items)

    return train_data, test_data, num_users, num_items


def deepcf_to_uir(train_file, test_file):
    """
    DeepCF 형식 파일을 Cornac UIR 튜플 형식으로 변환합니다.

    Args:
        train_file (str): train.rating 파일 경로
        test_file (str): test.rating 파일 경로

    Returns:
        list: (user, item, rating) 튜플의 리스트
    """
    data = []

    for file in [train_file, test_file]:
        with open(file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split('\t')
                    # Cornac을 위해 원본 ID를 문자열로 유지
                    data.append((parts[0], parts[1], float(parts[2])))

    return data


def get_train_matrix(train):
    """
    scipy 희소 행렬을 밀집 numpy 배열로 변환합니다.

    이 함수는 DMF, MLP, CFNet 모델에서 공유됩니다.
    희소 학습 데이터로부터 이진 상호작용 행렬을 생성합니다.

    Args:
        train (scipy.sparse.dok_matrix): 희소 학습 행렬

    Returns:
        np.ndarray: 밀집 이진 행렬 (num_users, num_items)
    """
    num_users, num_items = train.shape
    train_matrix = np.zeros([num_users, num_items], dtype=np.int32)
    for (u, i) in train.keys():
        train_matrix[u][i] = 1
    return train_matrix
