//==============================================================================
//
// Copyright 2018 The InsideLoop Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//==============================================================================

#ifndef IL_SPARSEMATRIXCSR_H
#define IL_SPARSEMATRIXCSR_H

#include <il/Array.h>
#include <il/Array2C.h>
#include <il/SmallArray.h>
#include <il/StaticArray.h>
#include <il/algorithmArray.h>
#include <il/linearAlgebra/dense/norm.h>
#include <il/math.h>

namespace il {

template <typename Index, typename T>
class SparseMatrixCSR {
 private:
  il::int_t n0_;
  il::int_t n1_;
  il::Array<T> element_;
  il::Array<Index> column_;
  il::Array<Index> row_;

 public:
  SparseMatrixCSR();
  SparseMatrixCSR(il::int_t height, il::int_t width, il::Array<Index> column,
                  il::Array<Index> row);
  SparseMatrixCSR(il::int_t height, il::int_t width, il::Array<Index> column,
                  il::Array<Index> row, il::Array<double> element);
  template <Index n>
  SparseMatrixCSR(il::int_t width, il::int_t height,
                  const il::Array<il::SmallArray<Index, n>> &column);
  SparseMatrixCSR(il::int_t n,
                  const il::Array<il::StaticArray<Index, 2>> &position,
                  il::io_t, il::Array<Index> &index);
  const T &operator[](il::int_t k) const;
  T &operator[](il::int_t k);
  const T &operator()(il::int_t i, il::int_t k) const;
  T &operator()(il::int_t i, il::int_t k);
  T element(il::int_t k) const;
  Index row(il::int_t i) const;
  Index column(il::int_t k) const;
  il::int_t size(il::int_t d) const;
  il::int_t nbNonZeros() const;
  const T *elementData() const;
  T *ElementData();
  const Index *rowData() const;
  Index *RowData();
  const Index *columnData() const;
  Index *ColumnData();
};

template <typename Index, typename T>
SparseMatrixCSR<Index, T>::SparseMatrixCSR() : element_{}, column_{}, row_{} {
  n0_ = 0;
  n1_ = 0;
}

template <typename Index, typename T>
SparseMatrixCSR<Index, T>::SparseMatrixCSR(il::int_t height, il::int_t width,
                                           il::Array<Index> column,
                                           il::Array<Index> row)
    : n0_{height},
      n1_{width},
      element_{column.size()},
      column_{std::move(column)},
      row_{std::move(row)} {
  IL_EXPECT_FAST(row_.size() == height + 1);
}

template <typename Index, typename T>
SparseMatrixCSR<Index, T>::SparseMatrixCSR(il::int_t height, il::int_t width,
                                           il::Array<Index> column,
                                           il::Array<Index> row,
                                           il::Array<double> element)
    : n0_{height},
      n1_{width},
      element_{std::move(element)},
      column_{std::move(column)},
      row_{std::move(row)} {
  IL_EXPECT_FAST(row_.size() == height + 1);
}

template <typename Index, typename T>
template <Index n>
SparseMatrixCSR<Index, T>::SparseMatrixCSR(
    il::int_t width, il::int_t height,
    const il::Array<il::SmallArray<Index, n>> &column)
    : n1_{width}, n0_{height}, element_{}, column_{}, row_{height + 1} {
  Index nb_nonzero = 0;
  for (Index i = 0; i < column.size(); ++i) {
    nb_nonzero += column[i].size();
  }

  element_.Resize(nb_nonzero);
  column_.Resize(nb_nonzero);
  row_[0] = 0;
  for (Index i = 0; i < column.size(); ++i) {
    for (Index k = 0; k < column[i].size(); ++k) {
      column_[row_[i] + k] = column[i][k];
    }
    row_[i + 1] = row_[i] + column[i].size();
  }
}

template <typename Index, typename T>
SparseMatrixCSR<Index, T>::SparseMatrixCSR(
    il::int_t n, const il::Array<il::StaticArray<Index, 2>> &position, il::io_t,
    il::Array<Index> &index)
    : element_{}, column_{}, row_{} {
  IL_EXPECT_FAST(n >= 0);
  //
  // element_
  // column_
  // row_
  n0_ = n;
  n1_ = n;

  const Index nb_entries = static_cast<Index>(position.size());
  index.Resize(nb_entries);

  // Compute the numbers of entries per Row. After this section, the
  // array nb_entries_per_Row will contains the numbers of entries in
  // the row i. The variable max_entries_per_row will contain the
  //  maximum number of entry in a row.
  il::Array<Index> nb_entries_per_row{n, 0};
  for (Index l = 0; l < nb_entries; ++l) {
    ++nb_entries_per_row[position[l][0]];
  }
  const Index max_entries_per_row = il::max(nb_entries_per_row);

  // Suppose that we have the matrix entered with the following entries:
  // (0, 0) - (2, 2) - (1, 2) - (1, 1) - (2, 2)
  // and the following values
  // a      - d      - c      - b      - e
  //
  // It would give us the following matrix
  // a 0 0
  // 0 b c
  // 0 0 d+e
  // There would be one entry in the first row, 2 entries in the second
  // one and 2 entries in the last one.
  // The integer entry_of_RowIndex(i, p) is set here to be the position
  // of the p-th entry of the row i in the list of entries of the matrix.
  // For instance entry_of_RowIndex(0, 0) = 0, entry_of_RowIndex(1, 0)=2,
  // entry_of_RowIndex(2, 1) = 4. The integer col_of_RowIndex is set
  // the same way and give the column number.
  // The array row_of_entry[l] gives us in which row the l-th entry is.
  // For instance row_of_entry[2] = 1
  // We first set those array without considering sorting the entries
  // of a line.
  il::Array2C<Index> entry_of_rowIndex{n, max_entries_per_row};
  il::Array2C<Index> col_of_rowIndex{n, max_entries_per_row};
  for (Index i = 0; i < n; ++i) {
    nb_entries_per_row[i] = 0;
  }
  for (Index l = 0; l < nb_entries; ++l) {
    Index i = position[l][0];
    Index p = nb_entries_per_row[i];
    entry_of_rowIndex(i, p) = l;
    col_of_rowIndex(i, p) = position[l][1];
    ++nb_entries_per_row[i];
  }
  // For each row, we sort them according to their column.
  for (Index i = 0; i < n; ++i) {
    for (Index p = 0; p < nb_entries_per_row[i] - 1; ++p) {
      Index min_col = static_cast<Index>(n);
      Index min_p = -1;
      for (Index q = p; q < nb_entries_per_row[i]; ++q) {
        if (col_of_rowIndex(i, q) < min_col) {
          min_col = col_of_rowIndex(i, q);
          min_p = q;
        }
      }
      Index tmp_entry = entry_of_rowIndex(i, p);
      Index tmp_col = col_of_rowIndex(i, p);
      entry_of_rowIndex(i, p) = entry_of_rowIndex(i, min_p);
      col_of_rowIndex(i, p) = col_of_rowIndex(i, min_p);
      entry_of_rowIndex(i, min_p) = tmp_entry;
      col_of_rowIndex(i, min_p) = tmp_col;
    };
  }

  // We count the number of non-zero elements per Row
  // which is less than the number of entries.
  il::Array<Index> nb_elements_per_row{n};
  Index nb_elements = 0;
  for (Index i = 0; i < n; ++i) {
    Index nb_elements_row = 0;
    Index last_newCol = -1;
    for (Index p = 0; p < nb_entries_per_row[i]; ++p) {
      if (col_of_rowIndex(i, p) != last_newCol) {
        last_newCol = col_of_rowIndex(i, p);
        ++nb_elements_row;
      }
    }
    nb_elements_per_row[i] = nb_elements_row;
    nb_elements += nb_elements_row;
  }

  // We then fill column_, row_
  element_.Resize(nb_elements);
  for (Index k = 0; k < element_.size(); ++k) {
    element_[k] = 0;
  }

  column_.Resize(nb_elements);
  row_.Resize(n + 1);
  row_[0] = 0;
  Index k = -1;
  for (Index i = 0; i < n; ++i) {
    Index last_newCol = -1;
    for (Index p = 0; p < nb_entries_per_row[i]; ++p) {
      if (col_of_rowIndex(i, p) != last_newCol) {
        ++k;
        last_newCol = col_of_rowIndex(i, p);
        column_[k] = col_of_rowIndex(i, p);
      }
      index[entry_of_rowIndex(i, p)] = k;
    }
    row_[i + 1] = row_[i] + nb_elements_per_row[i];
  }
}

template <typename Index, typename T>
T const &SparseMatrixCSR<Index, T>::operator[](il::int_t k) const {
  return element_[k];
}

template <typename Index, typename T>
T &SparseMatrixCSR<Index, T>::operator[](il::int_t k) {
  return element_[k];
}

template <typename Index, typename T>
T const &SparseMatrixCSR<Index, T>::operator()(il::int_t i, il::int_t k) const {
  IL_EXPECT_FAST(static_cast<std::size_t>(i) < static_cast<std::size_t>(n0_));
  IL_EXPECT_FAST(static_cast<std::size_t>(row_[i] + k) <
                 static_cast<std::size_t>(row_[i + 1]));
  return element_[row_[i] + k];
}

template <typename Index, typename T>
T &SparseMatrixCSR<Index, T>::operator()(il::int_t i, il::int_t k) {
  IL_EXPECT_FAST(static_cast<std::size_t>(i) < static_cast<std::size_t>(n0_));
  IL_EXPECT_FAST(static_cast<std::size_t>(row_[i] + k) <
                 static_cast<std::size_t>(row_[i + 1]));
  return element_[row_[i] + k];
}

template <typename Index, typename T>
il::int_t SparseMatrixCSR<Index, T>::size(il::int_t d) const {
  IL_EXPECT_MEDIUM(static_cast<std::size_t>(d) < static_cast<std::size_t>(2));
  return (d == 0) ? n0_ : n1_;
}

template <typename Index, typename T>
il::int_t SparseMatrixCSR<Index, T>::nbNonZeros() const {
  return element_.size();
}

template <typename Index, typename T>
const T *SparseMatrixCSR<Index, T>::elementData() const {
  return element_.data();
}

template <typename Index, typename T>
T *SparseMatrixCSR<Index, T>::ElementData() {
  return element_.Data();
}

template <typename Index, typename T>
const Index *SparseMatrixCSR<Index, T>::rowData() const {
  return row_.data();
}

template <typename Index, typename T>
Index *SparseMatrixCSR<Index, T>::RowData() {
  return row_.Data();
}

template <typename Index, typename T>
const Index *SparseMatrixCSR<Index, T>::columnData() const {
  return column_.data();
}

template <typename Index, typename T>
Index *SparseMatrixCSR<Index, T>::ColumnData() {
  return column_.Data();
}

template <typename Index, typename T>
T SparseMatrixCSR<Index, T>::element(il::int_t k) const {
  return element_[k];
}

template <typename Index, typename T>
Index SparseMatrixCSR<Index, T>::row(il::int_t i) const {
  return row_[i];
}

template <typename Index, typename T>
Index SparseMatrixCSR<Index, T>::column(il::int_t k) const {
  return column_[k];
}

template <typename Index>
inline double norm(const il::SparseMatrixCSR<Index, double> &A, Norm norm_type,
                   const il::Array<double> &beta,
                   const il::Array<double> &alpha) {
  IL_EXPECT_FAST(alpha.size() == A.size(0));
  IL_EXPECT_FAST(beta.size() == A.size(1));

  double norm = 0.0;
  switch (norm_type) {
    case Norm::Linf:
      for (Index i = 0; i < A.size(0); ++i) {
        double sum = 0.0;
        for (Index k = A.row(i); k < A.row(i + 1); ++k) {
          sum += il::abs(A[k] * alpha[A.column(k)] / beta[i]);
        }
        norm = il::max(norm, sum);
      }
      break;
    default:
      IL_EXPECT_FAST(false);
  }

  return norm;
}
}  // namespace il

#endif  // IL_SPARSEMATRIXCSR_H