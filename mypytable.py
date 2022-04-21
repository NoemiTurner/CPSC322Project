import copy
import csv
import pandas as pd


class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    """def pretty_print(self):
        Prints the table in a nicely formatted grid structure.
        
        print(tabulate(self.data, headers=self.column_names))"""

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        num_rows = len(self.data)
        num_cols = len(self.column_names)

        return num_rows, num_cols

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        # Include missing values
        try:
            col_index = self.column_names.index(col_identifier)
        except ValueError:
            raise ValueError("ValueError exception thrown")

        # Include missing values
        if(include_missing_values == True):
            col = []
            for row in self.data:
                value = row[col_index]
                col.append(value)
            return col
        else:  # Don't include missing values
            col = []
            for row in self.data:
                value = row[col_index]
                if value != "NA":
                    col.append(value)
            return col

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for values in self.data:
            for i in range(len(values)):
                try:
                    numeric_value = float(values[i])
                    # success
                    values[i] = numeric_value
                except ValueError:
                    exception = "exception"

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data. 

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        # Iterate through the table rows

        row_indexes_to_drop.sort(reverse=True)
        for index in row_indexes_to_drop:
            self.data.remove(self.data[index])

        return self

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        with open(filename, encoding="utf8") as f:
            csv_reader = csv.reader(f)
            for line_no, line in enumerate(csv_reader, 1):
                if line_no == 1:
                    self.column_names = line  # header
                else:
                    self.data.append(line)
        # convert numeric values
        self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        with open(filename, 'w', encoding='UTF8', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(self.column_names)
            csv_writer.writerows(self.data)
        return self

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        unique = []
        duplicates_indexes = []
        columns = []

        for row in self.data:
            temp = []
            for col in key_column_names:
                col_to_search = self.column_names.index(col)
                temp.append(row[col_to_search])
            columns.append(temp)

        row_num = 0
        for row in columns:
            if row not in unique:
                unique.append(row)
            else:
                duplicates_indexes.append(row_num)
            row_num += 1

        return duplicates_indexes

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        for row in self.data:
            if "NA" in row:
                self.data.remove(row)
        return self

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        column = self.get_column(col_name, False)
        sum_of_values = 0
        for value in column:
            sum_of_values = sum_of_values + value

        average = sum_of_values/len(column)

        for r, row in enumerate(self.data):
            for v, value in enumerate(row):
                if value == 'NA':
                    self.data[r][v] = average
        return self

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """

        data = []  # store the data that will be put into the new MyPyTable
        self.convert_to_numeric()

        for name in col_names:  # go through the col_names list
            column = self.get_column(name, False)  # get the column

            if(len(column) == 0):
                header = [["attribute"], ["min"], ["max"],
                          ["mid"], ["avg"], ["median"]]  # header
                data = []
                new_MyPyTable = MyPyTable(header, data)
                return new_MyPyTable

            # compute the average of the column
            average = sum(column)/len(column)
            min_val = min(column)  # compute the min value of the column
            max_val = max(column)  # compute the max value of the column
            # compute the mid values of the column
            mid_val = (max_val + min_val) / 2

            # compute the median value of the column
            middle = len(column) // 2
            column.sort()
            if len(column) % 2 == 0:  # if even
                median1 = column[len(column)//2]
                median2 = column[len(column)//2 - 1]
                median_val = (median1 + median2)/2
            else:  # if odd
                median_val = column[middle]

            data.append([name, min_val, max_val, mid_val, average, median_val])

        header = [["attribute"], ["min"], ["max"],
                  ["mid"], ["avg"], ["median"]]  # header
        new_MyPyTable = MyPyTable(header, data)

        return new_MyPyTable

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """

        df_left = pd.DataFrame(self.data, columns=self.column_names)
        df_right = pd.DataFrame(
            other_table.data, columns=other_table.column_names)
        df_joined = df_left.merge(df_right, how="inner", on=key_column_names)

        df_joined.fillna("NA", inplace=True)

        columns = []
        for c in self.column_names:
            if c not in columns:
                columns.append(c)
        for c in other_table.column_names:
            if c not in columns:
                columns.append(c)

        table = MyPyTable(columns, df_joined.values.tolist())

        return table

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """

        df_left = pd.DataFrame(self.data, columns=self.column_names)
        df_right = pd.DataFrame(
            other_table.data, columns=other_table.column_names)
        df_joined = df_left.merge(df_right, how="outer", on=key_column_names)

        df_joined.fillna("NA", inplace=True)

        columns = []
        for c in self.column_names:
            if c not in columns:
                columns.append(c)
        for c in other_table.column_names:
            if c not in columns:
                columns.append(c)

        table = MyPyTable(columns, df_joined.values.tolist())

        if c not in columns:
            columns.append(c)

        table = MyPyTable(columns, df_joined.values.tolist())

        return table
