class SparseMatrix:

    # Constructor to initialize a sparse matrix
    # Allows specifying number of rows and cols, sets them to 0 by default
    def __init__(self, rows=0, cols=0):
        self.data = {} 
        self.rows = rows 
        self.cols = cols

    # Set value at a given row and column,Checks for valid row/col, clears value if 0, updates rows/cols
    def set(self, row, col, value):
        if value < 0:
            raise ValueError("Value cannot be negative")
        if row < 0 or col < 0: 
            raise ValueError("Row and column indices must be non-negative")
        if value == 0 and (row, col) in self.data:
            del self.data[(row, col)] # Remove 0 values
        else:
            self.data[(row, col)] = value  
            self.rows = max(self.rows, row + 1)
            self.cols = max(self.cols, col + 1)

    # Get value at given row and column, returns 0 if none set
    def get(self, row, col):
        return self.data.get((row, col), 0)

    # Recommend items based on user vector multiplied by matrix values, raises error if dimensions mismatch
    def recommend(self, user_vector):
        if self.cols == 0:  
            return [0] * self.rows  
       
        # Check that vector length matches number of cols
        if self.cols != len(user_vector):
            raise ValueError("Matrix cols != length of user vector")

        recommendations = [0] * self.rows
        for row in range(self.rows):
            for col in range(self.cols):
                recommendations[row] += self.get(row, col) * user_vector[col]
        return recommendations

    # Checks that dimensions match first, Add two matrices together element-wise 
    def add_movie(self, matrix):
        if self.rows != matrix.rows or self.cols != matrix.cols:
            raise ValueError("Matrix dimensions must match")

        result = SparseMatrix()
        for row in range(self.rows):
            for col in range(self.cols):
                value = self.get(row, col) + matrix.get(row, col)
                if value != 0:
                    result.set(row, col, value)

        return result
    
    # Convert to dense matrix representation
    def to_dense(self):
        dense_matrix = [[0] * self.cols for _ in range(self.rows)]
        for (row, col), value in self.data.items():
            dense_matrix[row][col] = value
        return dense_matrix