import pytest
from sparse_recommender import SparseMatrix

def test_set_get():
    matrix = SparseMatrix()
    matrix.set(0, 0, 5)
    assert matrix.get(0, 0) == 5
    assert matrix.get(1, 1) == 0  # Ensure that unassigned values are 0
    assert matrix.get(2, 2) == 0

def test_set_zero_get():     
    matrix = SparseMatrix()
    matrix.set(1, 1, 0)
    assert matrix.get(1, 1) == 0

def test_recommend():
    matrix = SparseMatrix()
    matrix.set(0, 0, 5)
    matrix.set(1, 1, 3)
    vector = [2, 0]
    result = matrix.recommend(vector)
    assert result == [10, 0]

def test_recommend_with_error():
    matrix = SparseMatrix()
    matrix.set(0, 0, 5)
    matrix.set(1, 1, 3)
    vector = [2, 0, 1, 4]  # Incompatible vector with 4 elements
    with pytest.raises(ValueError):
        matrix.recommend(vector)

def test_add_movie():
    matrix1 = SparseMatrix()
    matrix1.set(0, 0, 3)
    matrix1.set(0, 1, 5)
    matrix2 = SparseMatrix()
    matrix2.set(0, 0, 6)
    matrix2.set(0, 1, 2)
    result = matrix1.add_movie(matrix2)
    assert result.get(0, 0) == 9
    assert result.get(0, 1) == 7

def test_to_dense():
    matrix = SparseMatrix()
    matrix.set(0, 0, 5)
    matrix.set(1, 1, 3)
    matrix.set(2, 2, 1)
    dense_matrix = matrix.to_dense()
    assert dense_matrix == [[5, 0, 0], [0, 3, 0], [0, 0, 1]]
    
## Additional test cases for edge cases and error handling

def test_set_negative_row_col():
    matrix = SparseMatrix()
    with pytest.raises(ValueError):
        matrix.set(-1, 2, 5)

# Test adding matrices of different sizes
def test_add_different_sizes():
    matrix1 = SparseMatrix(3, 4)
    matrix2 = SparseMatrix(3, 5)
    with pytest.raises(ValueError):
        matrix1.add_movie(matrix2)

def test_add_movie_invalid_dimensions_check():
    matrix1 = SparseMatrix()
    matrix1.set(0, 1, 5)
    matrix1.to_dense()
    matrix2 = SparseMatrix()
    matrix2.set(1, 1, 2)
    with pytest.raises(ValueError):
        matrix1.add_movie(matrix2)

def test_invalid_row_and_col():
    matrix = SparseMatrix()
    with pytest.raises(ValueError):
        matrix.set(-1,0,21)
    with pytest.raises(ValueError):
        matrix.set(0,-1,42)

def test_set_negative_value():
    matrix = SparseMatrix()
    with pytest.raises(ValueError):
        matrix.set(1, 1, -2)

def test_get_nonexistent_entry():
    matrix = SparseMatrix()
    assert matrix.get(2, 2) == 0

def test_recommend_empty_matrix():
    matrix = SparseMatrix()
    vector = [1, 2, 3]  # The user vector should match the number of columns in the matrix, but we don't have any columns yet in the empty matrix
    result = matrix.recommend(vector)
    assert result == []  # The expected result when the matrix has no values is an empty list

def test_to_dense_empty_matrix():
    matrix = SparseMatrix()
    dense_matrix = matrix.to_dense()
    assert dense_matrix == []  