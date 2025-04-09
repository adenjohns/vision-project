#include <iostream>
#include <Eigen/Dense>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;

/**
 * @brief std::tuple<int, int, int> find_largest_gap(VectorXi data_indices)
 *
 * Finds the largest traversable gap by counting the largest number of consecutive
 * numbers (indices) as well as their starting and ending indeces. The gaps in data_indices occurs from
 * reset_closest_points() which sets the points greater than the threshold to 0, and returns the indices
 * of the rest of the non-zero values in the array.
 *
 * @param data_indices a 1D array of numbers that are the indices of all the furthest data points for the find the gap algorithm.
 * @return max_length, max_start_idx, and max_end_idx.
        The max length is the length of the widest gap that exists.
        The max_start_idx is the starting index of the largest gap.
        The max_end_idx isthe ending index of the largest gap.
 */
std::tuple<int, int, int> find_largest_gap(VectorXd data_indices)
{
    int max_length = 1;
    int current_length = 1;

    int start_idx = 0;
    int max_start_idx = 0;
    int max_end_idx = 0;

    for (int i = 1; i < data_indices.size(); ++i)
    {
        if (data_indices(i) - data_indices(i - 1) == 1)
        { // Checking for increasing order
            current_length += 1;
        }
        else
        {
            if (current_length > max_length)
            {                                // If current length is greater than max length
                max_length = current_length; // Set max length to current length
                max_start_idx = start_idx;
                max_end_idx = i - 1;
            }

            current_length = 1; // Reset count
            start_idx = i;      // Update start index
        }
    }

    // Final check of the last value
    if (current_length > max_length)
    {
        max_length = current_length;
        max_start_idx = start_idx;
        max_end_idx = data_indices.size() - 1;
    }

    return std::make_tuple(max_length, max_start_idx, max_end_idx);
}

/**
 * @brief min_data_rows(MatrixXd array, int row_start, int row_end)
 *
 * Sets data greater than threshold to zero and returns the indices of all data points larger than zero.
 *
 * @param data The 1D array that needs to be parsed.
 * @param threshold An experimental value that is the limit to how close an object should be distance wise.
 * @return void. 1D array of all the indices of the data points larger than zero.
 */
void reset_closest_points(VectorXd data, int threshold, VectorXd& dataIndices)
{
    int count = 0;

    // Iterate through the vector and apply thresholding
    for (int i = 0; i < data.size(); ++i)
    {
        if (data(i) > threshold)
        {
            data(i) = 0; // Set values greater than the threshold to zero
        }
        else if (data(i) > 0)
        {
            // Store indices of values that are still nonzero after thresholding
            dataIndices(count++) = i; // Store index
        }
    }

    // Resize output vector to match the actual number of nonzero elements
    dataIndices.head(count);
}

/**
 * @brief min_data_rows(MatrixXd array, int row_start, int row_end)
 *
 * Takes a 2D array and first slices it based on the necessary rows needed given by
 * row_start and row_end, and then finds the min value of all the columns for n rows between the
 * given starting and ending rows.
 *
 * @param array The 2D array of distance data.
 * @param row_start An experimental value of the beginning of the rows that need to be parsed.
 * @param row_end An experimental value of the end of the rows that need to be parsed.
 * @return void. A 1D array of the min distance out of each col between row_start and row_end.
 */
void min_data_rows(MatrixXd array, int row_start, int row_end, int window_size, VectorXd& col_avg_val)
{
    int num_rows = row_end - row_start;
    int num_cols = array.cols();
    
    // Extract the submatrix (slicing rows from the given start row to the end row)
    MatrixXd sliced_arr = array.block(row_start, 0, num_rows, num_cols);
    MatrixXd smoothed = MatrixXd::Zero(num_rows - window_size + 1, num_cols); 
    
    for (int i = 0; i <= num_rows - window_size; ++i){
        smoothed.row(i) = sliced_arr.block(i, 0, window_size, num_cols).colwise().mean();
    }

    // Compute average along each column    
    col_avg_val = smoothed.colwise().mean();
    
    cout << col_avg_val << endl;
}

int main()
{
    MatrixXd m(5,10);
    m << 1, 1, 3, 4, 5, 2, 6, 6, 7, 4,
         1, 2, 4, 2, 5, 3, 1, 6, 7, 4,
         1, 1, 4, 2, 5, 3, 6, 6, 7, 4,
         1, 1, 4, 2, 5, 3, 1, 6, 7, 4,
         1, 1, 3, 4, 5, 2, 3, 6, 7, 4;
         
    int threshold = 5; // EXPERIMENTAL VALUE, depth values of object at closest limit to user 2999
    int row_start = 1;   // EXPERIMANTAL VALUE, depth value to first row from frame to parse 60 
    int row_end = 5;    // EXPERIMENTAL VALUE, depth value of last row from frame to parse 120
    int window_size = 2;
    
    VectorXd col_max_val(m.cols()); // Vector output for min_data_rows(). A 1D array of the min distance out of each col between row_start and row_end.
    min_data_rows(m, row_start, row_end, window_size, col_max_val);

    VectorXd data_indices(col_max_val.size()); // Vector output for reset_closet_points(). 1D array of all the indices of the data points larger than zero.
    reset_closest_points(col_max_val, threshold, data_indices);
    
    auto [max_length, max_start_idx, max_end_idx] = find_largest_gap(data_indices);

    cout << "Max Length: " << max_length << " Max Start Index: " << max_start_idx << " Max End Index: " << max_end_idx << endl;

    return 0;
}
