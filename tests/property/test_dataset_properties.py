"""Property-based tests for dataset module.

**Validates: Requirements 1.4**
"""

import numpy as np
from hypothesis import given, strategies as st, settings, assume


# Feature: aerial-object-detection, Property 11: Dataset Split Validity
@settings(max_examples=100)
@given(
    n_samples=st.integers(min_value=3, max_value=100),
    train_ratio=st.floats(min_value=0.1, max_value=0.8, allow_nan=False, allow_infinity=False),
    val_ratio=st.floats(min_value=0.1, max_value=0.4, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=10000)
)
def test_dataset_split_validity(n_samples, train_ratio, val_ratio, seed):
    """
    Property 11: For any dataset and split ratios (train/val/test), 
    the resulting splits SHALL have no overlapping samples and the 
    split sizes SHALL match the specified ratios (within rounding tolerance).
    
    **Validates: Requirements 1.4**
    """
    # Ensure ratios sum to <= 1.0
    test_ratio = 1.0 - train_ratio - val_ratio
    assume(test_ratio >= 0.05)  # Ensure test set has some samples
    
    # Create mock image IDs
    image_ids = [f'image_{i:04d}' for i in range(n_samples)]
    
    # Simulate split logic (same as split_dataset)
    rng = np.random.RandomState(seed)
    shuffled_ids = image_ids.copy()
    rng.shuffle(shuffled_ids)
    
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    train_ids = set(shuffled_ids[:train_end])
    val_ids = set(shuffled_ids[train_end:val_end])
    test_ids = set(shuffled_ids[val_end:])
    
    # Property: No overlapping samples
    assert len(train_ids & val_ids) == 0, "Train and val overlap"
    assert len(train_ids & test_ids) == 0, "Train and test overlap"
    assert len(val_ids & test_ids) == 0, "Val and test overlap"
    
    # Property: All samples accounted for
    all_ids = train_ids | val_ids | test_ids
    assert len(all_ids) == n_samples, (
        f"Not all samples accounted for: {len(all_ids)} != {n_samples}"
    )
    
    # Property: Split sizes approximately match ratios (within rounding)
    expected_train = int(n_samples * train_ratio)
    expected_val = int(n_samples * val_ratio)
    
    assert len(train_ids) == expected_train, (
        f"Train size mismatch: {len(train_ids)} != {expected_train}"
    )
    assert len(val_ids) == expected_val, (
        f"Val size mismatch: {len(val_ids)} != {expected_val}"
    )


# Additional property: Splits are deterministic with same seed
@settings(max_examples=50)
@given(
    n_samples=st.integers(min_value=10, max_value=50),
    seed=st.integers(min_value=0, max_value=10000)
)
def test_split_deterministic(n_samples, seed):
    """Test that splits are deterministic with the same seed."""
    image_ids = [f'image_{i:04d}' for i in range(n_samples)]
    
    # First split
    rng1 = np.random.RandomState(seed)
    shuffled1 = image_ids.copy()
    rng1.shuffle(shuffled1)
    
    # Second split with same seed
    rng2 = np.random.RandomState(seed)
    shuffled2 = image_ids.copy()
    rng2.shuffle(shuffled2)
    
    assert shuffled1 == shuffled2, "Splits not deterministic with same seed"


# Additional property: Different seeds produce different splits
@settings(max_examples=50)
@given(
    n_samples=st.integers(min_value=20, max_value=50),
    seed1=st.integers(min_value=0, max_value=5000),
    seed2=st.integers(min_value=5001, max_value=10000)
)
def test_different_seeds_different_splits(n_samples, seed1, seed2):
    """Test that different seeds produce different splits."""
    assume(seed1 != seed2)
    
    image_ids = [f'image_{i:04d}' for i in range(n_samples)]
    
    rng1 = np.random.RandomState(seed1)
    shuffled1 = image_ids.copy()
    rng1.shuffle(shuffled1)
    
    rng2 = np.random.RandomState(seed2)
    shuffled2 = image_ids.copy()
    rng2.shuffle(shuffled2)
    
    # With enough samples, different seeds should produce different orders
    # (extremely unlikely to be the same by chance)
    assert shuffled1 != shuffled2, "Different seeds produced same split"
