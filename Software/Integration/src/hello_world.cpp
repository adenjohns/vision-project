#include <iostream>
#include <vector> 
#include <algorithm> 
#include <Eigen/Dense>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;
using std::sort;

struct Gap {
    int start;                                                                              // the starting value of the gap
    int end;                                                                                // the ending value of the gap
    int length() const {return end - start + 1;}                                            // function that returns the length of the gap
};

bool compareGapLength(const Gap& a, const Gap& b)
{
    return a.length() > b.length();                                                         // longer gaps first
}

/**
 * @brief Find largest gap(s). 
 *
 * Finds the largest traversable gap by counting the largest number of consecutive
 * numbers (indices) as well as their starting and ending indices. The gaps in data_indices (breaks in consecutive numbers) occurs from
 * reset_closest_points() which sets the points greater than the threshold to 0, and returns the indices
 * of the rest of the non-zero values in the array.
 *
 * @param data_indices : A 1D array of numbers that are the indices of all the furthest data points for the find the gap algorithm.
 * @param topN : The number of gap vectors to record.
 * @return allGaps : A vector of all the gaps.
        start : The start of the gaps.
        end : The end of the gaps. 
        length : The total length of a gap.
 */
std::vector<Gap> find_largest_gaps(VectorXd data_indices, int topN = 2)
{ 
    vector<int> data_vec(data_indices.data(), data_indices.data() + data_indices.size());    // copies Eigen vector to normal vector 
    vector<Gap> allGaps;                                                                     // place to store all the existing gaps
    
    int start = data_vec[0];
    for (int i = 1; i < data_vec.size(); ++i)                                                // iterate through the indices of data 
    { 
        if (data_vec[i] != data_vec[i - 1] + 1)                                              // if the current index is not consecutive with the previous one, the gap ended
        {
            if (start != data_vec[i - 1])                                                    // if start and previous index are not the same, save the gap
            {
                allGaps.push_back({start, data_vec[i - 1]});                                 // creates a gap from the start to just before the current number 
            }
            start = data_vec[i];                                                             // start tracking next potential gap
        }
    }
    
    if (start != data_vec.back())                                                            // Track potential last gap in range
    {
        allGaps.push_back({start, data_vec.back()});
    }
    
    sort(allGaps.begin(), allGaps.end(), compareGapLength);                                  // Sorts gaps by length, from laragest to smallest
    
    if (allGaps.size() > static_cast<size_t>(topN))                                          // Shrink the vector to keep only the top N largest gaps
    {
        allGaps.resize(topN);
    }
    
    return allGaps;
}

/**
 * @brief Resets closes points and returns indices of rest. 
 *
 * Sets data greater than threshold to zero and returns the indices of all data points larger than zero.
 *
 * @param data : The 1D array that needs to be parsed.
 * @param threshold : An experimental value that is the limit to how close an object should be distance wise.
 * @return void : 1D array of all the indices of the data points larger than zero is copied into dataIndices.
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
 * @brief Returns the average distances of select rows per column. 
 *
 * Takes a 2D array and first slices it based on the necessary rows needed given by
 * row_start and row_end, and then finds the average value of all the columns for n rows between the
 * given starting and ending rows.
 *
 * @param array : The 2D array of distance data.
 * @param row_start : An experimental value of the beginning of the rows that need to be parsed.
 * @param row_end : An experimental value of the end of the rows that need to be parsed.
 * @return void : A 1D array of the min distance out of each col between row_start and row_end is copied into col_avg_val.
 */
void min_data_rows(MatrixXd array, int row_start, int row_end, VectorXd& col_avg_val)
{
    int num_rows = row_end - row_start;
    int num_cols = array.cols();
    
    // Extract the submatrix (slicing rows from the given start row to the end row)
    MatrixXd sliced_arr = array.block(row_start, 0, num_rows, num_cols);
    
    // Compute average along each column    
    col_avg_val = sliced_arr.colwise().mean();
    
    // MOVING AVERAGE IMPLEMENTATION (may be too slow for our needs)
    // // New matrix to store values of averages 
    // MatrixXd smoothed = MatrixXd::Zero(num_rows - window_size + 1, num_cols); 
    // for (int i = 0; i <= num_rows - window_size; ++i){
    //     smoothed.row(i) = sliced_arr.block(i, 0, window_size, num_cols).colwise().mean();
    // }
    // // Compute average along each column    
    // col_avg_val = smoothed.colwise().mean();
    
    // cout << col_avg_val << endl;
}

int main()
{
    MatrixXd m(5,10);
    m << 1, 1, 3, 4, 5, 2, 6, 6, 7, 4,
         1, 2, 4, 2, 5, 3, 1, 6, 7, 4,
         1, 1, 4, 2, 5, 3, 6, 6, 7, 4,
         1, 1, 4, 2, 5, 3, 1, 6, 7, 4,
         1, 1, 3, 4, 5, 2, 3, 6, 7, 4;
         
    int threshold = 5;   // EXPERIMENTAL VALUE, depth values of object at closest limit to user 2999
    int row_start = 1;   // EXPERIMANTAL VALUE, depth value to first row from frame to parse 60 
    int row_end = 5;     // EXPERIMENTAL VALUE, depth value of last row from frame to parse 120
    
    VectorXd col_max_val(m.cols());                                     // Vector output for min_data_rows(). A 1D array of the min distance out of each col between row_start and row_end.
    min_data_rows(m, row_start, row_end, col_max_val);

    VectorXd data_indices(col_max_val.size());                          // Vector output for reset_closet_points(). 1D array of all the indices of the data points larger than zero.
    reset_closest_points(col_max_val, threshold, data_indices);
    
    auto gaps = find_largest_gaps(data_indices, 2);
    
    if (gaps.empty()) 
    { 
        cout << "No gaps found.\n"; 
    }
    else 
    {
        for (size_t i = 0; i < gaps.size(); ++i) 
        {
            cout << "Gap " << i + 1 
                 << " -> Start: " << gaps[i].start << " End: " << gaps[i].end << ""
                 << ", Length: " << gaps[i].length() << "\n";
        }
    }

    return 0;
}
